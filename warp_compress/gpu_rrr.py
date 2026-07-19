"""gpu_rrr — an RRR succinct bitvector with rank on the GPU. Entropy-sized bitplanes, still O(1) rank.

The GPU wavelet (`gpu_wavelet.py`) stores bitplanes PACKED — n bits per level, exactly the raw index size.
RRR (Raman–Raman–Rao) stores each length-`T` block as (class, offset): the class = its popcount, the offset
= its **enumerative rank** among all T-bit words with that popcount. Skewed planes (which the BWT's are —
that's what takes the FM-index below H₀ toward Hₖ) then cost far less than 1 bit/bit, and rank is still O(1):
a superblock sample + a short in-block scan + a single **combinatorial decode of one block, in registers**.

That in-block decode is the whole trick — and it's what makes RRR "GPU-hostile" folklore wrong here: with a
small block (T=15) the binomial table is 16×16 and the unranking is a bounded loop a Warp thread runs fine.

    build: pack (class stream @4 bits, variable-width offset stream, superblock samples)
    rank1(i) on GPU: superblock jump → sum classes in-block → decode target block → popcount low bits

Standalone + validated here; wiring it under the wavelet's `_rank1` is the follow-up. Run:
python -m warp_compress.gpu_rrr
"""
from __future__ import annotations

import numpy as np
import warp as wp

from .gpu_wavelet import _popcount

wp.init()

T = 15                                       # block size (bits): C(15,k) ≤ 6435 ⇒ offsets fit in 13 bits
S = 64                                        # blocks per superblock — sparse rank samples (else the samples
                                             # dominate the size); a thread scans ≤S-1 blocks, cheap on GPU
_BINOM = np.zeros((T + 1, T + 1), np.int64)  # Pascal's triangle
for _nn in range(T + 1):
    _BINOM[_nn, 0] = 1
    for _kk in range(1, _nn + 1):
        _BINOM[_nn, _kk] = _BINOM[_nn - 1, _kk - 1] + _BINOM[_nn - 1, _kk]
_WIDTH = np.array([max(0, int(np.ceil(np.log2(_BINOM[T, k]))) if _BINOM[T, k] > 1 else 0)
                   for k in range(T + 1)], np.int32)   # offset bit-width per class


@wp.func
def _decode_word(binom: wp.array2d(dtype=wp.int32), cl: int, off: int) -> wp.uint32:
    """Combinatorial unrank: the T-bit word that has `cl` ones and enumerative rank `off`. In registers."""
    word = wp.uint32(0)
    r = off
    i = cl
    while i >= 1:
        c = T - 1
        while binom[c, i] > r:                # largest c with C(c,i) <= r
            c = c - 1
        word = word | (wp.uint32(1) << wp.uint32(c))
        r = r - binom[c, i]
        i = i - 1
    return word


@wp.func
def _classat(classes: wp.array(dtype=wp.uint32), j: int) -> int:
    bitpos = j * 4                            # 4 bits/class, never spans a word (4 | 32)
    return int((classes[bitpos >> 5] >> wp.uint32(bitpos & 31)) & wp.uint32(15))


@wp.func
def _readbits(stream: wp.array(dtype=wp.uint32), bitpos: int, width: int) -> int:
    if width == 0:
        return 0
    wi = bitpos >> 5
    b = bitpos & 31
    val = stream[wi] >> wp.uint32(b)
    if b + width > 32:
        val = val | (stream[wi + 1] << wp.uint32(32 - b))
    mask = (wp.uint32(1) << wp.uint32(width)) - wp.uint32(1)
    return int(val & mask)


@wp.kernel
def _rank1_k(classes: wp.array(dtype=wp.uint32), offsets: wp.array(dtype=wp.uint32),
             sbrank: wp.array(dtype=wp.int32), sboff: wp.array(dtype=wp.int32),
             width: wp.array(dtype=wp.int32), binom: wp.array2d(dtype=wp.int32),
             pos_in: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.int32)):
    t = wp.tid()
    pos = pos_in[t]
    blk = pos // T
    b = pos % T
    sbi = blk // S
    r = sbrank[sbi]
    obit = sboff[sbi]
    j = sbi * S
    while j < blk:                            # sum the classes (and offset widths) of the in-superblock blocks
        cl = _classat(classes, j)
        r = r + cl
        obit = obit + width[cl]
        j = j + 1
    if b > 0:                                 # partial popcount inside the target block
        cl = _classat(classes, blk)
        off = _readbits(offsets, obit, width[cl])
        word = _decode_word(binom, cl, off)
        mask = (wp.uint32(1) << wp.uint32(b)) - wp.uint32(1)
        r = r + _popcount(word & mask)
    out[t] = r


def rrr_encode(bits) -> dict:
    """Encode a 0/1 vector into RRR components (host arrays): packed class stream (4 bits/block), variable-
    width enumerative offset stream, and superblock rank/offset samples. Shared by GPURRR and the RRR wavelet.
    ``nblocks``/``nsb``/``cwords`` depend only on n, so every wavelet level has identical class/superblock
    shapes — only the offset stream length varies."""
    bits = (np.asarray(bits) != 0).astype(np.uint8)
    n = int(bits.shape[0])
    nblocks = (n + T - 1) // T
    pad = np.zeros(nblocks * T, np.uint8)
    pad[:n] = bits
    b2 = pad.reshape(nblocks, T)                                      # (nblocks, T) block bit matrix

    classes = b2.sum(1).astype(np.int64)                             # popcount per block
    cols = np.arange(T)[None, :]
    i_incl = np.cumsum(b2, axis=1)                                    # inclusive one-count up to each column
    offsets = (_BINOM[cols, i_incl] * b2).sum(1).astype(np.uint64)   # enumerative rank Σ C(p, i)

    cwords = (nblocks * 4 + 31) // 32 + 1                             # classes: 4 bits each
    cpk = np.zeros(cwords, np.uint32)
    bp = np.arange(nblocks) * 4
    np.bitwise_or.at(cpk, bp >> 5, (classes.astype(np.uint32) & 15) << (bp & 31).astype(np.uint32))

    wblk = _WIDTH[classes]                                            # offsets: variable width, bit-concatenated
    ostart = np.concatenate([[0], np.cumsum(wblk)]).astype(np.int64)
    total = int(ostart[-1])
    owords = (total + 31) // 32 + 2
    opk = np.zeros(owords, np.uint32)
    lw = ostart[:-1] >> 5
    lb = (ostart[:-1] & 31).astype(np.uint64)
    np.bitwise_or.at(opk, lw, ((offsets << lb) & 0xFFFFFFFF).astype(np.uint32))
    spill = (lb + wblk) > 32
    if spill.any():
        np.bitwise_or.at(opk, lw[spill] + 1, (offsets[spill] >> (32 - lb[spill])).astype(np.uint32))

    nsb = (nblocks + S - 1) // S                                      # superblock samples
    cum_class = np.concatenate([[0], np.cumsum(classes)]).astype(np.int64)
    sidx = np.minimum(np.arange(nsb + 1) * S, nblocks)
    return dict(cpk=cpk, opk=opk, sbrank=cum_class[sidx].astype(np.int32), sboff=ostart[sidx].astype(np.int32),
                nblocks=nblocks, nsb=nsb, cwords=cwords, class_bits=nblocks * 4, offset_bits=total)


class GPURRR:
    """RRR succinct bitvector resident on a Warp device; batched rank1 runs entirely on the GPU."""

    def __init__(self, bits, device: str = "cuda:0"):
        self.n = int(np.asarray(bits).shape[0])
        self.device = device
        e = rrr_encode(bits)
        self._nblocks, self._class_bits, self._offset_bits = e["nblocks"], e["class_bits"], e["offset_bits"]
        self._sb_bytes = e["sbrank"].nbytes + e["sboff"].nbytes
        self.classes = wp.array(e["cpk"], dtype=wp.uint32, device=device)
        self.offsets = wp.array(e["opk"], dtype=wp.uint32, device=device)
        self.sbrank = wp.array(e["sbrank"], dtype=wp.int32, device=device)
        self.sboff = wp.array(e["sboff"], dtype=wp.int32, device=device)
        self.width = wp.array(_WIDTH, dtype=wp.int32, device=device)
        self.binom = wp.array(_BINOM.astype(np.int32), dtype=wp.int32, device=device)

    def size_bits(self) -> int:
        """Stored bits: class stream + offset stream + superblock samples (tables are tiny & shared)."""
        return self._class_bits + self._offset_bits + self._sb_bytes * 8

    def rank1(self, positions) -> np.ndarray:
        pos = wp.array(np.asarray(positions, np.int32), dtype=wp.int32, device=self.device)
        out = wp.zeros(pos.shape[0], dtype=wp.int32, device=self.device)
        wp.launch(_rank1_k, dim=pos.shape[0],
                  inputs=[self.classes, self.offsets, self.sbrank, self.sboff, self.width, self.binom,
                          pos, out], device=self.device)
        wp.synchronize_device(self.device)
        return out.numpy()


def _demo():
    import math

    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    rng = np.random.default_rng(0)
    n = 2_000_000
    print(f"device={dev}   N={n:,}   block T={T}, superblock S={S}")
    print(f"  {'density p':>10} {'H0(p)':>7} {'RRR b/bit':>10} {'packed':>7} {'vs packed':>10}  rank✓")
    for p in (0.5, 0.1, 0.03, 0.005):
        bits = (rng.random(n) < p).astype(np.uint8)
        rr = GPURRR(bits, device=dev)
        # correctness: batched GPU rank1 vs the naive prefix popcount
        cum = np.concatenate([[0], np.cumsum(bits)])
        qpos = rng.integers(0, n + 1, 20000).astype(np.int32)
        got = rr.rank1(qpos)
        ok = np.array_equal(got, cum[qpos])
        h0 = -(p * math.log2(p) + (1 - p) * math.log2(1 - p)) if 0 < p < 1 else 0.0
        bpb = rr.size_bits() / n
        print(f"  {p:>10.3f} {h0:>7.3f} {bpb:>10.3f} {1.0:>7.3f} {1.0/bpb:>9.2f}×  {'✓' if ok else 'FAIL'}")
    print("=> skewed planes cost far below 1 bit/bit and approach H0 — while rank1 stays O(1) and runs on the "
          "GPU (a per-thread superblock jump + one in-register combinatorial block decode). This is the lever\n"
          "   that pulls the resident FM-index from packed (1 b/bit) toward the entropy Hk. Next: wire it under "
          "the wavelet's _rank1 so the whole self-index is entropy-sized AND GPU-searchable.")


if __name__ == "__main__":
    _demo()
