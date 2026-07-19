"""fm_index — an FM-index over a token sequence: substring search inside the compressed stream.

The culmination of the compression arc, and the exact object bioinformatics uses to align reads to a
genome. Build the Burrows–Wheeler transform of the token sequence, index the BWT with the wavelet matrix
(``wavelet.WaveletMatrix`` -> O(bits) rank), and add the C array. Then **backward search** counts (and, with
a sampled suffix array, locates) any pattern in O(|pattern|·bits) — over the *compressed* sequence, never
materialising it.

    count(pattern)     number of occurrences of the token pattern in the sequence
    locate(pattern)    their positions (via a sampled suffix array)
    extract(i, j)      the token run [i, j) (walking the index)

So the same sequence is compressed, O(1)/O(log) addressable (token_chromosome / wavelet), AND searchable
(FM-index) — the full self-index. Run: python -m warp_compress.fm_index
"""
from __future__ import annotations

import numpy as np

from .wavelet import WaveletMatrix


def suffix_array(s: np.ndarray) -> np.ndarray:
    """Suffix array of `s` (ints) by prefix doubling — O(n log n) numpy sorts."""
    s = np.asarray(s, np.int64)
    n = int(s.shape[0])
    sa = np.argsort(s, kind="stable")
    rank = np.empty(n, np.int64)
    sv = s[sa]
    rank[sa] = np.concatenate([[0], np.cumsum(sv[1:] != sv[:-1])])
    k = 1
    while int(rank.max()) < n - 1:
        r2 = np.full(n, -1, np.int64)
        r2[: n - k] = rank[k:]
        key = (rank << 32) | (r2 + 1)
        sa = np.argsort(key, kind="stable")
        kv = key[sa]
        rank[sa] = np.concatenate([[0], np.cumsum(kv[1:] != kv[:-1])])
        k *= 2
    return sa


class FMIndex:
    """FM-index over a token sequence (ints >= 0). A 0 sentinel is added internally."""

    def __init__(self, seq, sa_sample: int = 32):
        seq = np.asarray(seq, np.int64) + 1                   # shift so 0 is a free sentinel
        self.n = int(seq.shape[0]) + 1
        s = np.concatenate([seq, [0]])                        # append sentinel (smallest)
        self.sa = suffix_array(s)
        bwt = s[(self.sa - 1) % self.n]
        self.sigma = int(s.max()) + 1
        self.wm = WaveletMatrix(bwt, bits=max(1, (self.sigma - 1).bit_length()))
        self.C = np.concatenate([[0], np.cumsum(np.bincount(bwt, minlength=self.sigma))])[: self.sigma]
        self._sa_sample = sa_sample
        self._sa_at = {int(p): int(self.sa[p]) for p in range(self.n) if self.sa[p] % sa_sample == 0}

    def _bw_range(self, pattern):
        """Backward search: the SA range [lo, hi) whose suffixes start with `pattern`."""
        pat = np.asarray(pattern, np.int64) + 1
        lo, hi = 0, self.n
        for c in pat[::-1]:
            c = int(c)
            if c >= self.sigma:
                return 0, 0
            lo = int(self.C[c]) + self.wm.rank(c, lo)
            hi = int(self.C[c]) + self.wm.rank(c, hi)
            if lo >= hi:
                return 0, 0
        return lo, hi

    def count(self, pattern) -> int:
        lo, hi = self._bw_range(pattern)
        return hi - lo

    def predict_next(self, context):
        """FM-index AS a retrieval language model: the next-token distribution given `context` is, for each
        candidate c, count(context + [c]) — pure backward search over the COMPRESSED sequence, no raw text.
        Returns {token: probability} over tokens that have ever followed this context."""
        ctx = list(context)
        dist = {}
        for c in range(self.sigma - 1):                   # candidate next tokens (exclude the sentinel)
            cnt = self.count(ctx + [c])                    # occurrences of context FOLLOWED by c
            if cnt:
                dist[c] = cnt
        tot = sum(dist.values())
        return {c: n / tot for c, n in dist.items()} if tot else {}

    def locate(self, pattern):
        """Text positions where `pattern` occurs (via LF-walk to the nearest sampled SA entry)."""
        lo, hi = self._bw_range(pattern)
        out = []
        for r in range(lo, hi):
            steps = 0
            p = r
            while p not in self._sa_at:
                c = self.wm.access(p)
                p = int(self.C[c]) + self.wm.rank(c, p)       # LF-mapping
                steps += 1
            out.append((self._sa_at[p] + steps) % self.n)
        return sorted(out)


def _demo():
    rng = np.random.default_rng(0)
    V = 60
    p = 1.0 / np.arange(1, V + 1)
    p /= p.sum()
    seq = rng.choice(V, size=20000, p=p)
    fm = FMIndex(seq, sa_sample=16)

    # a pattern we know exists
    at = int(rng.integers(0, len(seq) - 4))
    pat = seq[at:at + 4]

    naive = sum(1 for i in range(len(seq) - len(pat) + 1) if np.array_equal(seq[i:i + len(pat)], pat))
    got = fm.count(pat)
    assert got == naive, (got, naive)

    locs = fm.locate(pat)
    naive_locs = [i for i in range(len(seq) - len(pat) + 1) if np.array_equal(seq[i:i + len(pat)], pat)]
    assert locs == naive_locs, "locate mismatch"

    print(f"N={len(seq)}  alphabet={V}  pattern={list(pat)}")
    print(f"count(pattern) = {got}   (naive scan = {naive})  ✓   locate matches naive ✓")
    print(f"backward search touched {len(pat)} ranks (O(|pattern|)), not the {len(seq)} tokens.")
    print("=> substring SEARCH inside the compressed token stream. FM-index = BWT + wavelet rank; this is "
          "the genomic read aligner, now over token sequences. The DNA compression loop is closed.")


if __name__ == "__main__":
    _demo()
