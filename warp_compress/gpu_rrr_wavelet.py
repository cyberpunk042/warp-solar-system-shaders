"""gpu_rrr_wavelet — the wavelet matrix backed by RRR, on the GPU. The whole self-index, entropy-sized.

`gpu_wavelet.GPUWavelet` stores each level PACKED (n bits). This stores each level as an **RRR** bitvector
(`gpu_rrr`) instead, so the resident index shrinks toward the sequence's entropy — and because the FM-index
indexes the **BWT** (whose bitplanes are skewed by construction), that means toward Hₖ — while `access`/`rank`
still run entirely on the GPU. One object: entropy-sized *and* GPU-searchable.

Layout note: n is the same at every level, so `nblocks`/`nsb`/`cwords` are identical across levels ⇒ the class
stream and superblock samples stack as clean 2-D arrays `[level, ·]`; only the variable-width **offset** stream
differs per level, so those are concatenated flat with a per-level word base. Run:
python -m warp_compress.gpu_rrr_wavelet
"""
from __future__ import annotations

import numpy as np
import warp as wp

from .gpu_rrr import S, T, _BINOM, _WIDTH, _decode_word, rrr_encode
from .gpu_wavelet import _popcount


@wp.func
def _classat2(classes: wp.array2d(dtype=wp.uint32), lvl: int, j: int) -> int:
    bp = j * 4
    return int((classes[lvl, bp >> 5] >> wp.uint32(bp & 31)) & wp.uint32(15))


@wp.func
def _readbits2(off: wp.array(dtype=wp.uint32), bitpos: int, width: int) -> int:
    if width == 0:
        return 0
    wi = bitpos >> 5
    b = bitpos & 31
    val = off[wi] >> wp.uint32(b)
    if b + width > 32:
        val = val | (off[wi + 1] << wp.uint32(32 - b))
    mask = (wp.uint32(1) << wp.uint32(width)) - wp.uint32(1)
    return int(val & mask)


@wp.func
def _rank1_lvl(classes: wp.array2d(dtype=wp.uint32), offsets: wp.array(dtype=wp.uint32),
               sbrank: wp.array2d(dtype=wp.int32), sboff: wp.array2d(dtype=wp.int32),
               offbase: wp.array(dtype=wp.int32), width: wp.array(dtype=wp.int32),
               binom: wp.array2d(dtype=wp.int32), lvl: int, pos: int) -> int:
    """RRR rank1 for one wavelet level: superblock jump + in-block class scan + one register block decode."""
    blk = pos // T
    b = pos % T
    sbi = blk // S
    r = sbrank[lvl, sbi]
    obit = sboff[lvl, sbi]
    j = sbi * S
    while j < blk:
        cl = _classat2(classes, lvl, j)
        r = r + cl
        obit = obit + width[cl]
        j = j + 1
    if b > 0:
        cl = _classat2(classes, lvl, blk)
        o = _readbits2(offsets, offbase[lvl] + obit, width[cl])
        word = _decode_word(binom, cl, o)
        mask = (wp.uint32(1) << wp.uint32(b)) - wp.uint32(1)
        r = r + _popcount(word & mask)
    return r


@wp.kernel
def _access_k(classes: wp.array2d(dtype=wp.uint32), offsets: wp.array(dtype=wp.uint32),
              sbrank: wp.array2d(dtype=wp.int32), sboff: wp.array2d(dtype=wp.int32),
              offbase: wp.array(dtype=wp.int32), zeros: wp.array(dtype=wp.int32),
              width: wp.array(dtype=wp.int32), binom: wp.array2d(dtype=wp.int32),
              pos_in: wp.array(dtype=wp.int32), tok_out: wp.array(dtype=wp.int32), bits: int):
    t = wp.tid()
    i = pos_in[t]
    v = int(0)
    for lvl in range(bits):
        r0 = _rank1_lvl(classes, offsets, sbrank, sboff, offbase, width, binom, lvl, i)
        r1 = _rank1_lvl(classes, offsets, sbrank, sboff, offbase, width, binom, lvl, i + 1)
        if r1 - r0 == 1:
            v = (v << 1) | 1
            i = zeros[lvl] + r0
        else:
            v = v << 1
            i = i - r0
    tok_out[t] = v


@wp.kernel
def _rank_k(classes: wp.array2d(dtype=wp.uint32), offsets: wp.array(dtype=wp.uint32),
            sbrank: wp.array2d(dtype=wp.int32), sboff: wp.array2d(dtype=wp.int32),
            offbase: wp.array(dtype=wp.int32), zeros: wp.array(dtype=wp.int32),
            width: wp.array(dtype=wp.int32), binom: wp.array2d(dtype=wp.int32),
            c_in: wp.array(dtype=wp.int32), i_in: wp.array(dtype=wp.int32),
            out: wp.array(dtype=wp.int32), bits: int):
    t = wp.tid()
    c = c_in[t]
    p = int(0)
    q = i_in[t]
    for lvl in range(bits):
        bitc = (c >> (bits - 1 - lvl)) & 1
        rp = _rank1_lvl(classes, offsets, sbrank, sboff, offbase, width, binom, lvl, p)
        rq = _rank1_lvl(classes, offsets, sbrank, sboff, offbase, width, binom, lvl, q)
        if bitc == 1:
            p = zeros[lvl] + rp
            q = zeros[lvl] + rq
        else:
            p = p - rp
            q = q - rq
    out[t] = q - p


class RRRWaveletGPU:
    """Wavelet matrix whose every level is an RRR bitvector, resident on a Warp device. access/rank on GPU."""

    def __init__(self, seq, device: str = "cuda:0", bits: int | None = None):
        seq = np.asarray(seq, np.int64)
        self.n = int(seq.shape[0])
        self.bits = int(bits) if bits is not None else max(1, int(seq.max()).bit_length())
        self.device = device

        cpks, sbranks, sboffs, opks, zeros = [], [], [], [], []
        offbase = [0]
        self._bits_stored = 0
        cur = seq.copy()
        for lvl in range(self.bits):
            b = ((cur >> (self.bits - 1 - lvl)) & 1).astype(np.uint8)
            e = rrr_encode(b)
            cpks.append(e["cpk"]); sbranks.append(e["sbrank"]); sboffs.append(e["sboff"]); opks.append(e["opk"])
            zeros.append(self.n - int(b.sum()))
            offbase.append(offbase[-1] + e["opk"].shape[0])
            self._bits_stored += e["class_bits"] + e["offset_bits"]
            order = np.concatenate([np.flatnonzero(b == 0), np.flatnonzero(b == 1)])
            cur = cur[order]

        classes = np.stack(cpks)                                     # (bits, cwords)
        sbrank = np.stack(sbranks)                                   # (bits, nsb+1)
        sboff = np.stack(sboffs)
        offsets = np.concatenate(opks)                               # flat, per-level word base in `offbase`
        self._sb_bytes = sbrank.nbytes + sboff.nbytes
        self.classes = wp.array(classes, dtype=wp.uint32, device=device)
        self.offsets = wp.array(offsets, dtype=wp.uint32, device=device)
        self.sbrank = wp.array(sbrank, dtype=wp.int32, device=device)
        self.sboff = wp.array(sboff, dtype=wp.int32, device=device)
        self.offbase = wp.array(np.asarray(offbase[:-1], np.int32) * 32, dtype=wp.int32, device=device)
        self.zeros = wp.array(np.asarray(zeros, np.int32), dtype=wp.int32, device=device)
        self.width = wp.array(_WIDTH, dtype=wp.int32, device=device)
        self.binom = wp.array(_BINOM.astype(np.int32), dtype=wp.int32, device=device)

    def index_bytes(self) -> int:
        """Resident footprint: RRR class + offset streams (all levels) + superblock samples."""
        return self._bits_stored // 8 + self._sb_bytes

    def _args(self):
        return [self.classes, self.offsets, self.sbrank, self.sboff, self.offbase, self.zeros,
                self.width, self.binom]

    def access(self, positions) -> np.ndarray:
        pos = wp.array(np.asarray(positions, np.int32), dtype=wp.int32, device=self.device)
        out = wp.zeros(pos.shape[0], dtype=wp.int32, device=self.device)
        wp.launch(_access_k, dim=pos.shape[0], inputs=[*self._args(), pos, out, self.bits], device=self.device)
        wp.synchronize_device(self.device)
        return out.numpy()

    def rank(self, symbols, positions) -> np.ndarray:
        c = wp.array(np.asarray(symbols, np.int32), dtype=wp.int32, device=self.device)
        i = wp.array(np.asarray(positions, np.int32), dtype=wp.int32, device=self.device)
        out = wp.zeros(c.shape[0], dtype=wp.int32, device=self.device)
        wp.launch(_rank_k, dim=c.shape[0], inputs=[*self._args(), c, i, out, self.bits], device=self.device)
        wp.synchronize_device(self.device)
        return out.numpy()


@wp.func
def _wrank_rrr(classes: wp.array2d(dtype=wp.uint32), offsets: wp.array(dtype=wp.uint32),
               sbrank: wp.array2d(dtype=wp.int32), sboff: wp.array2d(dtype=wp.int32),
               offbase: wp.array(dtype=wp.int32), zeros: wp.array(dtype=wp.int32),
               width: wp.array(dtype=wp.int32), binom: wp.array2d(dtype=wp.int32),
               c: int, i: int, bits: int) -> int:
    """Full wavelet rank(c, i) over the RRR-backed levels."""
    p = int(0)
    q = i
    for lvl in range(bits):
        bitc = (c >> (bits - 1 - lvl)) & 1
        rp = _rank1_lvl(classes, offsets, sbrank, sboff, offbase, width, binom, lvl, p)
        rq = _rank1_lvl(classes, offsets, sbrank, sboff, offbase, width, binom, lvl, q)
        if bitc == 1:
            p = zeros[lvl] + rp
            q = zeros[lvl] + rq
        else:
            p = p - rp
            q = q - rq
    return q - p


@wp.kernel
def _bw_rrr_k(classes: wp.array2d(dtype=wp.uint32), offsets: wp.array(dtype=wp.uint32),
              sbrank: wp.array2d(dtype=wp.int32), sboff: wp.array2d(dtype=wp.int32),
              offbase: wp.array(dtype=wp.int32), zeros: wp.array(dtype=wp.int32),
              width: wp.array(dtype=wp.int32), binom: wp.array2d(dtype=wp.int32),
              C: wp.array(dtype=wp.int32), pat: wp.array(dtype=wp.int32),
              starts: wp.array(dtype=wp.int32), lens: wp.array(dtype=wp.int32),
              out: wp.array(dtype=wp.int32), bits: int, sigma: int, n: int):
    t = wp.tid()
    st = starts[t]
    L = lens[t]
    lo = int(0)
    hi = n
    for k in range(L):
        c = pat[st + L - 1 - k]
        if c < sigma:
            if lo < hi:
                lo = C[c] + _wrank_rrr(classes, offsets, sbrank, sboff, offbase, zeros, width, binom, c, lo, bits)
                hi = C[c] + _wrank_rrr(classes, offsets, sbrank, sboff, offbase, zeros, width, binom, c, hi, bits)
        else:
            lo = int(0)
            hi = int(0)
    out[t] = hi - lo


class GPURRRFMIndex:
    """An FM-index whose BWT wavelet is RRR-compressed AND GPU-searchable: entropy-sized count/predict_next."""

    def __init__(self, seq, device: str = "cuda:0"):
        from .fm_index import suffix_array

        seq = np.asarray(seq, np.int64) + 1
        self.n = int(seq.shape[0]) + 1
        s = np.concatenate([seq, [0]])
        bwt = s[(suffix_array(s) - 1) % self.n]
        self.sigma = int(s.max()) + 1
        self.device = device
        self.wm = RRRWaveletGPU(bwt, device=device, bits=max(1, (self.sigma - 1).bit_length()))
        self.bits = self.wm.bits
        C = np.concatenate([[0], np.cumsum(np.bincount(bwt, minlength=self.sigma))])[: self.sigma]
        self.C = wp.array(C.astype(np.int32), dtype=wp.int32, device=device)

    def index_bytes(self) -> int:
        return self.wm.index_bytes()

    def count(self, patterns) -> np.ndarray:
        flat, starts, lens = [], [], []
        for p in patterns:
            starts.append(len(flat))
            lens.append(len(p))
            flat.extend(int(x) + 1 for x in p)
        pat = wp.array(np.asarray(flat or [0], np.int32), dtype=wp.int32, device=self.device)
        st = wp.array(np.asarray(starts, np.int32), dtype=wp.int32, device=self.device)
        ln = wp.array(np.asarray(lens, np.int32), dtype=wp.int32, device=self.device)
        out = wp.zeros(len(patterns), dtype=wp.int32, device=self.device)
        wp.launch(_bw_rrr_k, dim=len(patterns),
                  inputs=[*self.wm._args(), self.C, pat, st, ln, out, self.bits, self.sigma, self.n],
                  device=self.device)
        wp.synchronize_device(self.device)
        return out.numpy()

    def predict_next(self, context, vocab: int | None = None) -> np.ndarray:
        V = int(vocab) if vocab is not None else self.sigma - 1
        counts = self.count([list(context) + [c] for c in range(V)]).astype(np.float64)
        tot = counts.sum()
        return counts / tot if tot > 0 else counts


def _demo():
    import math

    from .fm_index import suffix_array
    from .gpu_wavelet import GPUWavelet

    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    rng = np.random.default_rng(0)
    # a STRUCTURED stream (Markov) -> its BWT has skewed bitplanes -> RRR wins. This is the FM-index's world.
    V = 64
    trans = rng.dirichlet(np.ones(V) * 0.3, size=V)
    n = 2_000_000
    seq = np.empty(n, np.int64)
    seq[0] = 0
    for i in range(1, n):
        seq[i] = rng.choice(V, p=trans[seq[i - 1]])

    s = np.concatenate([seq + 1, [0]])
    bwt = s[(suffix_array(s) - 1) % s.shape[0]]                       # index the BWT, like the FM-index

    rrw = RRRWaveletGPU(bwt, device=dev)
    pk = GPUWavelet(bwt, device=dev)

    # correctness: access reconstructs the BWT; rank matches naive
    qpos = rng.integers(0, bwt.shape[0], 20000).astype(np.int32)
    assert np.array_equal(rrw.access(qpos), bwt[qpos]), "RRR-wavelet access mismatch"
    qc = rng.integers(0, V + 1, 3000).astype(np.int32)
    qi = rng.integers(0, bwt.shape[0] + 1, 3000).astype(np.int32)
    gr = rrw.rank(qc, qi)
    for j in rng.integers(0, 3000, 10):
        assert gr[j] == int(np.count_nonzero(bwt[: qi[j]] == qc[j])), "RRR-wavelet rank mismatch"

    h0 = -sum((cnt / bwt.shape[0]) * math.log2(cnt / bwt.shape[0]) for cnt in np.bincount(bwt) if cnt)
    print(f"device={dev}   BWT length={bwt.shape[0]:,}   alphabet<=2^{rrw.bits}")
    print(f"[correct] RRR-wavelet access == BWT ✓   rank == naive ✓")
    print(f"[size]  packed wavelet  {pk.index_bytes()/1e6:6.2f} MB ({pk.index_bytes()*8/bwt.shape[0]:.2f} b/tok)")
    print(f"        RRR   wavelet  {rrw.index_bytes()/1e6:6.2f} MB ({rrw.index_bytes()*8/bwt.shape[0]:.2f} b/tok)"
          f"   => {pk.index_bytes()/rrw.index_bytes():.2f}× smaller, still GPU access/rank")
    print(f"        (H0 of the BWT = {h0:.2f} b/tok; the FM-index reaches toward Hk below that)")

    # the entropy-sized index is genuinely searchable: count/predict_next over the RRR-backed FM-index
    from .fm_index import FMIndex
    gfm = GPURRRFMIndex(seq, device=dev)
    cfm = FMIndex(seq[:200_000])
    pats = [[int(x) for x in seq[a:a + 3]] for a in rng.integers(0, n - 3, 100)]
    gc = gfm.count(pats)
    ok_count = all(gc[j] == sum(1 for i in range(n - 3) if list(seq[i:i + 3]) == pats[j])
                   for j in rng.integers(0, 100, 4))
    ctx = [int(seq[123]), int(seq[124])]
    gtop = int(np.argmax(GPURRRFMIndex(seq[:200_000], device=dev).predict_next(ctx, vocab=V)))
    ctop = max(cfm.predict_next(ctx), key=cfm.predict_next(ctx).get)
    print(f"[search] entropy-sized FM-index: GPU count == naive {'✓' if ok_count else 'FAIL'}   "
          f"predict_next top-1 == CPU {'✓' if gtop == ctop else 'FAIL'}")
    print("=> the SAME object is now entropy-sized AND GPU-searchable: every wavelet level is an RRR bitvector "
          "decoded in registers, so access/rank AND FM-index count/predict_next run on the compact index.")


if __name__ == "__main__":
    _demo()
