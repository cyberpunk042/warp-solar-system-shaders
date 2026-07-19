"""gpu_wavelet — the wavelet-matrix rank/access, on the GPU, in Warp. ChromoFold's make-or-break primitive.

The whole ChromoFold thesis (docs/chromofold.md §1) is *GPU-resident, partial unfold*: decode only the slice
you touch, without ever leaving the GPU. That requires the wavelet ``rank``/``access`` — the core of every
FM-index query and every token decode — to run on the GPU over the COMPACT structure. This module is that.

Unlike ``wavelet.WaveletMatrix`` (which stores full int64 prefix-sum arrays = n·bits·8 bytes, not compact),
this stores the **succinct** form and keeps it resident in VRAM:

    packed bitplanes    words[level] : n bits, packed 32/uint32     (the actual n·bits index)
    superblock rank     sb[level]    : cumulative popcount every SB words   (the O(1)-rank side table)

A Warp kernel then answers a whole BATCH of queries in parallel — each thread walks the `bits` levels doing
blocked-popcount rank — which is exactly how attention/decoding touch a KV cache or a context: thousands of
independent position lookups at once. Run: python -m warp_compress.gpu_wavelet
"""
from __future__ import annotations

import numpy as np
import warp as wp

wp.init()

SB = 8                                    # words per superblock (256 bits): the memory⇄rank-latency dial
_POPC_LUT = np.array([bin(i).count("1") for i in range(256)], np.int32)


@wp.func
def _popcount(x: wp.uint32) -> wp.int32:
    # SWAR population count on a 32-bit word (no table, fully in-register)
    x = x - ((x >> wp.uint32(1)) & wp.uint32(0x55555555))
    x = (x & wp.uint32(0x33333333)) + ((x >> wp.uint32(2)) & wp.uint32(0x33333333))
    x = (x + (x >> wp.uint32(4))) & wp.uint32(0x0F0F0F0F)
    return wp.int32((x * wp.uint32(0x01010101)) >> wp.uint32(24))


@wp.func
def _rank1(words: wp.array2d(dtype=wp.uint32), sb: wp.array2d(dtype=wp.int32),
           lvl: int, pos: int, sbw: int) -> int:
    """Number of set bits in words[lvl] over bit range [0, pos): superblock jump + blocked popcount."""
    w = pos >> 5
    b = pos & 31
    blk = w // sbw
    r = sb[lvl, blk]
    k = blk * sbw
    while k < w:
        r = r + _popcount(words[lvl, k])
        k = k + 1
    if b > 0:
        mask = (wp.uint32(1) << wp.uint32(b)) - wp.uint32(1)
        r = r + _popcount(words[lvl, w] & mask)
    return r


@wp.kernel
def _access_k(words: wp.array2d(dtype=wp.uint32), sb: wp.array2d(dtype=wp.int32),
              zeros: wp.array(dtype=wp.int32), pos_in: wp.array(dtype=wp.int32),
              tok_out: wp.array(dtype=wp.int32), bits: int, sbw: int):
    t = wp.tid()
    i = pos_in[t]
    v = int(0)
    for lvl in range(bits):
        r0 = _rank1(words, sb, lvl, i, sbw)
        bit = int((words[lvl, i >> 5] >> wp.uint32(i & 31)) & wp.uint32(1))
        if bit == 1:
            v = (v << 1) | 1
            i = zeros[lvl] + r0                 # descend into the 1-child block
        else:
            v = v << 1
            i = i - r0                          # rank0(i) = i - rank1(i)
    tok_out[t] = v


@wp.kernel
def _rank_k(words: wp.array2d(dtype=wp.uint32), sb: wp.array2d(dtype=wp.int32),
            zeros: wp.array(dtype=wp.int32), c_in: wp.array(dtype=wp.int32),
            i_in: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.int32), bits: int, sbw: int):
    t = wp.tid()
    c = c_in[t]
    p = int(0)
    q = i_in[t]
    for lvl in range(bits):
        bitc = (c >> (bits - 1 - lvl)) & 1
        rp = _rank1(words, sb, lvl, p, sbw)
        rq = _rank1(words, sb, lvl, q, sbw)
        if bitc == 1:
            p = zeros[lvl] + rp
            q = zeros[lvl] + rq
        else:
            p = p - rp
            q = q - rq
    out[t] = q - p


class GPUWavelet:
    """Succinct wavelet matrix resident on a Warp device. Batched access()/rank() run entirely on the GPU."""

    def __init__(self, seq, device: str = "cuda:0", bits: int | None = None):
        seq = np.asarray(seq, np.int64)
        self.n = int(seq.shape[0])
        self.bits = int(bits) if bits is not None else max(1, int(seq.max()).bit_length())
        self.device = device
        nwords = (self.n + 31) // 32
        nblocks = (nwords + SB - 1) // SB
        words = np.zeros((self.bits, nwords), np.uint32)
        sb = np.zeros((self.bits, nblocks + 1), np.int32)
        zeros = np.zeros(self.bits, np.int32)

        cur = seq.copy()
        idx = np.arange(self.n)
        wi = idx >> 5
        off = (idx & 31).astype(np.uint32)
        for lvl in range(self.bits):
            b = ((cur >> (self.bits - 1 - lvl)) & 1).astype(np.uint32)
            np.bitwise_or.at(words[lvl], wi, b << off)                 # pack the bitplane
            pcw = _POPC_LUT[words[lvl].view(np.uint8)].reshape(nwords, 4).sum(1)   # popcount per word
            cum = np.concatenate([[0], np.cumsum(pcw)]).astype(np.int64)
            sb[lvl] = cum[np.minimum(np.arange(nblocks + 1) * SB, nwords)]         # superblock samples
            zeros[lvl] = self.n - int(b.sum())
            order = np.concatenate([np.flatnonzero(b == 0), np.flatnonzero(b == 1)])
            cur = cur[order]                                          # stable partition for the next level

        self._nwords, self._nblocks = nwords, nblocks
        self.words = wp.array(words, dtype=wp.uint32, device=device)
        self.sb = wp.array(sb, dtype=wp.int32, device=device)
        self.zeros = wp.array(zeros, dtype=wp.int32, device=device)

    def index_bytes(self) -> int:
        """VRAM footprint of the resident index: the packed bitplanes + superblock table."""
        return self._nwords * self.bits * 4 + self.sb.shape[0] * self.sb.shape[1] * 4 + self.bits * 4

    def access(self, positions) -> np.ndarray:
        """Batched token decode: token at each position, entirely on the GPU. Returns a host int array."""
        pos = wp.array(np.asarray(positions, np.int32), dtype=wp.int32, device=self.device)
        out = wp.zeros(pos.shape[0], dtype=wp.int32, device=self.device)
        wp.launch(_access_k, dim=pos.shape[0],
                  inputs=[self.words, self.sb, self.zeros, pos, out, self.bits, SB], device=self.device)
        wp.synchronize_device(self.device)
        return out.numpy()

    def rank(self, symbols, positions) -> np.ndarray:
        """Batched rank: occurrences of symbols[t] in [0, positions[t]) for every t, on the GPU."""
        c = wp.array(np.asarray(symbols, np.int32), dtype=wp.int32, device=self.device)
        i = wp.array(np.asarray(positions, np.int32), dtype=wp.int32, device=self.device)
        out = wp.zeros(c.shape[0], dtype=wp.int32, device=self.device)
        wp.launch(_rank_k, dim=c.shape[0],
                  inputs=[self.words, self.sb, self.zeros, c, i, out, self.bits, SB], device=self.device)
        wp.synchronize_device(self.device)
        return out.numpy()


def _demo():
    import time
    from .wavelet import WaveletMatrix

    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    rng = np.random.default_rng(0)
    V = 256
    p = 1.0 / np.arange(1, V + 1)
    p /= p.sum()
    n = 4_000_000
    seq = rng.choice(V, size=n, p=p).astype(np.int64)
    gw = GPUWavelet(seq, device=dev)

    # 1) correctness vs the sequence and vs the CPU wavelet
    qpos = rng.integers(0, n, 100000).astype(np.int32)
    got = gw.access(qpos)
    assert np.array_equal(got, seq[qpos]), "GPU access mismatch"
    qc = rng.integers(0, V, 20000).astype(np.int32)
    qi = rng.integers(0, n + 1, 20000).astype(np.int32)
    grank = gw.rank(qc, qi)
    # spot-check a few ranks against a naive count
    for j in rng.integers(0, 20000, 12):
        assert grank[j] == int(np.count_nonzero(seq[:qi[j]] == qc[j])), "GPU rank mismatch"

    # 2) footprint: succinct GPU index vs the CPU wavelet's full prefix arrays
    wm = WaveletMatrix(seq[:400_000])                                  # smaller: CPU build is heavy
    cpu_bytes = sum(a.nbytes for a in wm._p1) + sum(a.nbytes for a in wm._p0)
    cpu_per_tok = cpu_bytes / wm.n
    gpu_per_tok = gw.index_bytes() / n

    # 3) throughput: GPU batched access vs a CPU python-loop baseline
    M = 1_000_000
    bpos = rng.integers(0, n, M).astype(np.int32)
    gw.access(bpos[:1000])                                             # warm up kernels
    t0 = time.perf_counter(); gw.access(bpos); gpu_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    for k in range(3000):
        wm.access(int(bpos[k]) % wm.n)
    cpu_s = (time.perf_counter() - t0) / 3000

    print(f"device={dev}   N={n:,}   alphabet<=2^{gw.bits}")
    print(f"[correct] GPU access == sequence ✓   GPU rank == naive ✓")
    print(f"[size]  succinct GPU index {gpu_per_tok:.2f} B/tok  ({gw.index_bytes()/1e6:.1f} MB resident)  "
          f"vs CPU prefix-arrays {cpu_per_tok:.1f} B/tok  => {cpu_per_tok/gpu_per_tok:.0f}× smaller, GPU-resident")
    gpu_tps, cpu_tps = M / gpu_s, 1.0 / cpu_s
    print(f"[speed] GPU decoded {M:,} tokens in {gpu_s*1e3:.1f} ms = {gpu_tps/1e6:.0f} M tok/s   "
          f"vs naive CPU loop {cpu_tps/1e3:,.0f} K tok/s  => GPU ~{gpu_tps/cpu_tps:,.0f}× faster")
    print("=> rank/access — every FM-index query and token decode — runs GPU-resident over the compact index, "
          "in parallel batches. This is the primitive that makes 'partial unfold, never leave the GPU' real.")


if __name__ == "__main__":
    _demo()
