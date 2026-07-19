"""wavelet — a wavelet matrix over a token sequence: the principled compressed self-index.

Where ``token_chromosome`` gives O(1) *positional* access (position -> token) via a space-filling curve,
a **wavelet matrix** (Claude & Navarro 2012, the practical form of the wavelet tree) gives the full
succinct-index toolkit over a sequence on an alphabet [0, 2**bits):

    access(i)     O(bits)  -> the token at position i
    rank(c, i)    O(bits)  -> how many c's occur in [0, i)
    select(c, k)  O(bits log n) -> the position of the k-th c (0-indexed)

rank/select are the primitives behind the **FM-index**: back the sequence by its BWT and these give
O(|pattern|) substring search inside the compressed text — literally how genomic reads are aligned. So the
DNA metaphor closes the loop: the compressed, addressable, *searchable* sequence index is the FM-index, and
this is its wavelet-matrix core, applied to token streams.

Size here is the raw n*bits bitplanes (same as fixed-width); compressing the bitplanes (RRR / gzip) with
rank support takes it to entropy — noted as the next step. Run: python -m warp_compress.wavelet
"""
from __future__ import annotations

import numpy as np


class WaveletMatrix:
    """Wavelet matrix over `seq` (ints in [0, 2**bits)). Supports access / rank / select."""

    def __init__(self, seq, bits: int | None = None):
        seq = np.asarray(seq, np.int64)
        self.n = int(seq.shape[0])
        self.bits = int(bits) if bits is not None else max(1, int(seq.max()).bit_length())
        self.zeros: list[int] = []
        self._p1: list[np.ndarray] = []      # prefix count of 1s per level, length n+1
        self._p0: list[np.ndarray] = []      # prefix count of 0s per level, length n+1
        cur = seq.copy()
        idx = np.arange(self.n + 1)
        for lvl in range(self.bits):
            b = ((cur >> (self.bits - 1 - lvl)) & 1).astype(np.int64)
            p1 = np.concatenate([[0], np.cumsum(b)])
            self._p1.append(p1)
            self._p0.append(idx - p1)
            self.zeros.append(int(self.n - int(p1[-1])))
            zi = np.flatnonzero(b == 0)
            oi = np.flatnonzero(b == 1)
            cur = cur[np.concatenate([zi, oi])]   # stable partition: 0s then 1s

    # --- rank/select on a single bitplane, O(1) / O(log n) via the prefix sums ---
    def _r1(self, lvl: int, i: int) -> int:
        return int(self._p1[lvl][i])

    def _r0(self, lvl: int, i: int) -> int:
        return int(self._p0[lvl][i])

    def _sel1(self, lvl: int, k: int) -> int:                 # position of the k-th 1 (1-indexed)
        return int(np.searchsorted(self._p1[lvl], k)) - 1

    def _sel0(self, lvl: int, k: int) -> int:
        return int(np.searchsorted(self._p0[lvl], k)) - 1

    # --- the three self-index operations ---
    def access(self, i: int) -> int:
        v = 0
        for lvl in range(self.bits):
            one = self._r1(lvl, i + 1) - self._r1(lvl, i)     # is bit set at position i?
            if one:
                v = (v << 1) | 1
                i = self.zeros[lvl] + self._r1(lvl, i)
            else:
                v = v << 1
                i = self._r0(lvl, i)
        return v

    def rank(self, c: int, i: int) -> int:
        """Number of occurrences of `c` in positions [0, i)."""
        p, q = 0, int(i)
        for lvl in range(self.bits):
            if (c >> (self.bits - 1 - lvl)) & 1:
                p = self.zeros[lvl] + self._r1(lvl, p)
                q = self.zeros[lvl] + self._r1(lvl, q)
            else:
                p = self._r0(lvl, p)
                q = self._r0(lvl, q)
        return q - p

    def select(self, c: int, k: int) -> int:
        """Position of the k-th occurrence of `c` (0-indexed k: k=0 is the first). -1 if fewer than k+1."""
        if k < 0 or self.rank(c, self.n) <= k:
            return -1
        p = 0
        bit_at = []
        for lvl in range(self.bits):
            one = (c >> (self.bits - 1 - lvl)) & 1
            bit_at.append((one, p))
            p = self.zeros[lvl] + self._r1(lvl, p) if one else self._r0(lvl, p)
        pos = p + k                                            # position within c's block at the bottom
        for lvl in range(self.bits - 1, -1, -1):
            one, base = bit_at[lvl]
            if one:
                pos = self._sel1(lvl, pos - self.zeros[lvl] + 1)
            else:
                pos = self._sel0(lvl, pos + 1)
        return int(pos)

    def size_bits(self) -> int:
        return self.n * self.bits                             # raw bitplanes (compressible to entropy)


def _demo():
    rng = np.random.default_rng(0)
    V = 256
    p = 1.0 / np.arange(1, V + 1)
    p /= p.sum()
    seq = rng.choice(V, size=50000, p=p)
    wm = WaveletMatrix(seq)

    # 1) access reconstructs the sequence exactly
    assert all(wm.access(i) == seq[i] for i in rng.integers(0, wm.n, 1000))

    # 2) rank matches the naive prefix count; select finds the k-th occurrence
    for c in rng.integers(0, V, 20):
        i = int(rng.integers(0, wm.n + 1))
        assert wm.rank(int(c), i) == int(np.count_nonzero(seq[:i] == c))
    for c in rng.integers(0, V, 20):
        occ = np.flatnonzero(seq == c)
        if occ.size:
            k = int(rng.integers(0, occ.size))
            assert wm.select(int(c), k) == int(occ[k])

    import math
    H0 = -sum((cnt / wm.n) * math.log2(cnt / wm.n)
              for cnt in np.bincount(seq) if cnt)              # zeroth-order entropy
    print(f"N={wm.n}  alphabet<=2^{wm.bits}  raw index = {wm.size_bits()/8/1e3:.1f} KB  "
          f"(H0 lower bound = {H0*wm.n/8/1e3:.1f} KB)")
    print("access / rank / select all verified against naive.")
    c = int(np.bincount(seq).argmax())
    print(f"e.g. token {c}: appears {wm.rank(c, wm.n)}x; 5th occurrence at position {wm.select(c, 4)}; "
          f"access(0)={wm.access(0)} (== seq[0]={seq[0]})")
    print("=> compressed self-index over the token stream: O(bits) access + rank/select. rank/select on a "
          "BWT = FM-index => O(|pattern|) substring search in the compressed sequence (the genomic aligner).")


if __name__ == "__main__":
    _demo()
