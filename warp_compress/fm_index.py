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

    def prob_next(self, context, c, max_order: int = 8, D: float = 4.0) -> float:
        """Interpolated variable-order backoff probability of token `c` given `context` — the '∞-gram' LM
        built straight from the index. Start at the unigram (add-1), interpolate UP through longer contexts;
        each order's weight lambda = count(ctx)/(count(ctx)+D) (Witten-Bell-ish: trust higher order when it
        has support). O(max_order) backward searches — no V loop, so it's cheap for perplexity."""
        c = int(c)
        p = (self.count([c]) + 1.0) / ((self.n - 1) + (self.sigma - 1))     # unigram, add-1
        for L in range(1, min(max_order, len(context)) + 1):
            ctx = list(context[-L:])
            n_ctx = self.count(ctx)
            if n_ctx == 0:
                continue                                                     # order unseen -> keep backoff
            lam = n_ctx / (n_ctx + D)
            p = lam * (self.count(ctx + [c]) / n_ctx) + (1.0 - lam) * p
        return p

    def distinct_following(self, context) -> int:
        """N1+(context·): number of distinct token TYPES that follow `context`."""
        ctx = list(context)
        return sum(1 for c in range(self.sigma - 1) if self.count(ctx + [c]) > 0)

    def distinct_preceding(self, c: int) -> int:
        """N1+(·c): number of distinct token TYPES that PRECEDE token c — cheap via the BWT, since the BWT
        column over c's suffix range IS the multiset of preceding symbols. This is KN's continuation count."""
        cc = int(c) + 1
        lo, hi = int(self.C[cc]), (int(self.C[cc + 1]) if cc + 1 < len(self.C) else self.n)
        return sum(1 for s in range(self.sigma)
                   if self.wm.rank(s, hi) - self.wm.rank(s, lo) > 0)

    def _total_distinct_bigrams(self) -> int:
        if not hasattr(self, "_ndb"):
            self._ndb = sum(self.distinct_preceding(c) for c in range(self.sigma - 1))
        return int(self._ndb)

    def prob_kn(self, context, c, max_order: int = 4, d: float = 0.75) -> float:
        """Interpolated Kneser-Ney from the index. Lowest order = the CONTINUATION probability
        N1+(·c)/N1+(··) (from the BWT), then absolute-discount up through longer contexts using
        N1+(ctx·) as the interpolation mass. Absolute discounting is the right fix for high-order overfit."""
        c = int(c)
        p = (self.distinct_preceding(c) + 1e-9) / max(self._total_distinct_bigrams(), 1)
        for L in range(1, min(max_order, len(context)) + 1):
            ctx = list(context[-L:])
            n_ctx = self.count(ctx)
            if n_ctx == 0:
                continue
            higher = max(self.count(ctx + [c]) - d, 0.0) / n_ctx
            lam = d * self.distinct_following(ctx) / n_ctx        # interpolation weight to the lower order
            p = higher + lam * p
        return p

    def _cont_probs(self) -> np.ndarray:
        """Cached KN continuation base P_cont(c) = N1+(·c)/N1+(··) for all tokens."""
        if not hasattr(self, "_cp"):
            v = self.sigma - 1
            n1 = np.array([self.distinct_preceding(c) for c in range(v)], np.float64)
            self._cp = (n1 + 1e-9) / max(n1.sum(), 1.0)
        return self._cp

    def next_dist_kn(self, context, max_order: int = 6, d: float = 0.75) -> np.ndarray:
        """Full next-token KN distribution over all tokens — O(max_order · V) (the interpolation mass and
        continuation base are per-context / cached, not per-candidate)."""
        v = self.sigma - 1
        p = self._cont_probs().copy()
        for L in range(1, min(max_order, len(context)) + 1):
            ctx = list(context[-L:])
            n_ctx = self.count(ctx)
            if n_ctx == 0:
                continue
            cnt = np.array([self.count(ctx + [c]) for c in range(v)], np.float64)   # O(V) searches
            lam = d * float((cnt > 0).sum()) / n_ctx                                 # N1+(ctx·)/n_ctx
            p = np.maximum(cnt - d, 0.0) / n_ctx + lam * p
        s = p.sum()
        return p / s if s > 0 else p

    def generate(self, prompt, length: int, max_order: int = 6, temperature: float = 0.8,
                 top_k: int = 0, seed: int = 0, d: float = 0.75, repeat_penalty: float = 1.0):
        """The index WRITES: sample `length` tokens, each from its own KN next-distribution. Training-free
        generation straight from the compressed self-index — the 'navigate it token by token' loop, closed.
        `repeat_penalty` > 1 down-weights the just-emitted token (tames char-n-gram whitespace/indent runs)."""
        rng = np.random.default_rng(seed)
        out = list(prompt)
        v = self.sigma - 1
        for _ in range(length):
            p = self.next_dist_kn(out, max_order, d)
            if temperature != 1.0:
                p = np.power(p, 1.0 / max(temperature, 1e-3))
            if repeat_penalty != 1.0 and out:
                p[out[-1] % v] /= repeat_penalty
            if top_k and top_k < v:
                cut = np.argpartition(p, -top_k)[:-top_k]
                p[cut] = 0.0
            s = p.sum()
            if s <= 0:
                out.append(int(rng.integers(v)))
                continue
            out.append(int(rng.choice(v, p=p / s)))
        return out

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

    # the index WRITES: sample tokens straight from its own KN next-distribution — no model, no training.
    import glob
    txt = "".join(open(f).read() for f in sorted(glob.glob("warp_compress/*.py")))[:120000]
    cseq = np.frombuffer(txt.encode("utf-8", "ignore"), np.uint8).astype(np.int64)
    cfm = FMIndex(cseq, sa_sample=64)
    gen = cfm.generate([int(b) for b in b"def "], 180, max_order=7,
                       temperature=0.55, top_k=10, seed=1, repeat_penalty=1.4)
    print(f"\ngenerate() over {len(cseq)} chars of repo source, seeded 'def ':")
    print("  " + bytes(min(c, 255) for c in gen).decode("utf-8", "replace").replace("\n", "\n  "))
    print("=> the compressed self-index composes text token by token. Same object that is compressed, "
          "addressable, and searchable is ALSO a generative LM. The 'navigate it token by token' loop is closed.")


if __name__ == "__main__":
    _demo()
