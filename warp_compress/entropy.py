"""entropy — measure the achievable compressed-index rates (#1 RRR to H0, #2 BWT to Hk).

- ``H0``/``Hk``: empirical zeroth- and k-th-order entropy of a token sequence (bits/symbol).
- ``rrr_bits``: the size of a bitvector under RRR (per-block popcount + enumerative offset) — the succinct
  structure that stores a bitvector in ~its entropy WHILE keeping O(1) rank. Summed over the wavelet
  bitplanes this reaches n*H0 with rank support (#1).
- ``index_report``: puts numbers on it. The FM-index indexes the BWT, whose H0 approaches the sequence's Hk
  (the BWT clusters equal contexts), so the FM-index compresses to ~n*Hk, not just n*H0 (#2).

Run: python -m warp_compress.entropy
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .fm_index import suffix_array
from .wavelet import WaveletMatrix


def H0(seq) -> float:
    seq = np.asarray(seq)
    n = int(seq.shape[0])
    if n == 0:
        return 0.0
    _, c = np.unique(seq, return_counts=True)
    p = c / n
    return float(-(p * np.log2(p)).sum())


def Hk(seq, k: int) -> float:
    """k-th order empirical entropy (bits/symbol): average H0 of the next symbol given the length-k context."""
    seq = np.asarray(seq)
    n = int(seq.shape[0])
    if k == 0 or n <= k:
        return H0(seq)
    ctx = defaultdict(list)
    for i in range(k, n):
        ctx[tuple(seq[i - k:i].tolist())].append(int(seq[i]))
    total = 0.0
    for nexts in ctx.values():
        a = np.asarray(nexts)
        _, c = np.unique(a, return_counts=True)
        p = c / a.shape[0]
        total += a.shape[0] * float(-(p * np.log2(p)).sum())
    return total / (n - k)


def rrr_bits(bits, block: int = 15) -> int:
    """Bits to store a 0/1 vector under RRR: per block, a popcount header + the enumerative index of the
    block among all patterns with that popcount (ceil(log2 C(block, popcount))). O(1) rank via samples
    (a small additive term, folded in). This is ~ the bitvector's H0, with rank support preserved."""
    bits = np.asarray(bits).astype(np.int64)
    n = int(bits.shape[0])
    hdr = math.ceil(math.log2(block + 1))
    total = 0
    for s in range(0, n, block):
        blk = bits[s:s + block]
        b = int(blk.shape[0])
        pc = int(blk.sum())
        total += hdr + (math.ceil(math.log2(math.comb(b, pc))) if 0 < pc < b else 0)
    samples = math.ceil(n / (block * 32)) * max(1, math.ceil(math.log2(max(n, 2))))   # O(1)-rank samples
    return total + samples


def _wavelet_index_bits(seq, rrr: bool) -> int:
    seq = np.asarray(seq, np.int64)
    bits = max(1, int(seq.max()).bit_length())
    n = int(seq.shape[0])
    total = 0
    cur = seq.copy()
    for lvl in range(bits):
        b = (cur >> (bits - 1 - lvl)) & 1
        total += rrr_bits(b) if rrr else n
        order = np.concatenate([np.flatnonzero(b == 0), np.flatnonzero(b == 1)])
        cur = cur[order]
    return total


def index_report(seq):
    seq = np.asarray(seq, np.int64)
    n = int(seq.shape[0])
    h0, h1, h2 = H0(seq), Hk(seq, 1), Hk(seq, 2)
    print(f"N={n}  V={len(np.unique(seq))}   H0={h0:.3f}  H1={h1:.3f}  H2={h2:.3f}  bits/symbol")

    raw = _wavelet_index_bits(seq, rrr=False)
    rrr = _wavelet_index_bits(seq, rrr=True)
    # FM-index indexes the BWT: it clusters equal contexts into RUNS, so its wavelet bitplanes become
    # skewed and RRR compresses them below H0, toward Hk (higher-order) -- while staying searchable.
    s = np.concatenate([seq + 1, [0]])
    bwt = s[(suffix_array(s) - 1) % s.shape[0]]
    bwt_rrr = _wavelet_index_bits(bwt, rrr=True)
    print(f"[#1] wavelet-of-sequence: raw {raw/8/1e3:5.1f} KB  RRR {rrr/8/1e3:5.1f} KB   "
          f"(n*H0={n*h0/8/1e3:.1f}, n*H2={n*h2/8/1e3:.1f} KB)")
    print(f"     RRR only helps when bitplanes are SKEWED; a balanced source barely moves (block overhead).")
    print(f"[#2] wavelet-of-BWT (FM-index): RRR {bwt_rrr/8/1e3:5.1f} KB  -> BELOW n*H0 ({n*h0/8/1e3:.1f}), "
          f"heading to n*Hk ({n*h2/8/1e3:.1f})")
    print(f"     the BWT's context-runs skew the bitplanes => higher-order compression, still searchable.")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # a structured stream: a small Markov source (real higher-order redundancy, so Hk << H0)
    V = 24
    trans = rng.dirichlet(np.ones(V) * 0.3, size=V)     # peaky next-symbol distributions
    seq = np.empty(60000, np.int64)
    seq[0] = 0
    for i in range(1, len(seq)):
        seq[i] = rng.choice(V, p=trans[seq[i - 1]])
    index_report(seq)
