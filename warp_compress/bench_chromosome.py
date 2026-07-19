"""Benchmark token_chromosome vs standard compressors — honest numbers on rate AND random access.

The chromosome's compression comes from the merge codec (dedup) + run-length structure; its structural win
is O(1)/O(log) **random access + local navigation**, which gzip/lzma cannot do (they need to decompress
from the start). This measures both, on three stream types:

  repetitive  — few types, long runs (merge-codec friendly, like the card)
  zipfian     — text-like: many types, skewed frequency
  random      — many types, uniform (near-incompressible)

Run:  python -m warp_compress.bench_chromosome
"""
from __future__ import annotations

import gzip
import lzma
import math
import time

import numpy as np

from .token_chromosome import compress


def _streams():
    rng = np.random.default_rng(0)
    out = {}
    blk = rng.integers(0, 32, 2500)
    out["repetitive"] = np.repeat(blk, rng.integers(4, 20, blk.shape))
    V = 2000
    p = 1.0 / np.arange(1, V + 1)
    p /= p.sum()
    out["zipfian"] = rng.choice(V, size=80000, p=p)
    out["random"] = rng.integers(0, 4000, 80000)
    return out


def _width(maxval: int) -> int:
    w = max(1, math.ceil(math.log2(max(maxval + 1, 2)) / 8))
    return {1: 1, 2: 2}.get(w, 4 if w <= 4 else 8)


def _pack(a: np.ndarray) -> bytes:
    return a.astype(f"<u{_width(int(a.max()) if a.size else 1)}").tobytes()


def _rle(ids: np.ndarray):
    change = np.flatnonzero(np.diff(ids)) + 1
    starts = np.concatenate([[0], change])
    vals = ids[starts].astype(np.int64)
    lengths = np.diff(np.concatenate([starts, [len(ids)]])).astype(np.int64)
    cum = np.concatenate([[0], np.cumsum(lengths)])
    return vals, lengths, cum


def bench():
    rng = np.random.default_rng(1)
    print(f"{'stream':<11} {'N':>7} {'V':>5} | {'raw':>8} {'gzip':>8} {'lzma':>8} "
          f"{'chromo':>8} {'chromo+gz':>9} | {'access chromo':>14} {'access gzip':>12}")
    print("-" * 108)
    for name, seq in _streams().items():
        ch = compress(seq, dim=3)
        book, ids = ch.book, ch.ids
        N, V = ch.n, int(book.shape[0])
        vals, lengths, cum = _rle(ids)

        raw = _pack(ids)
        gz = len(gzip.compress(raw, 6))
        xz = len(lzma.compress(raw))
        # chromosome storage = content book + RLE(symbols, lengths) [+ tiny map constants]
        chromo = book.nbytes + vals.nbytes + lengths.nbytes + 8
        chromo_gz = book.nbytes + len(gzip.compress(_pack(vals) + _pack(lengths), 6)) + 8

        # --- random access: chromosome (binary-search the RLE) vs gzip (must decompress the stream) ---
        K = 5000
        qs = rng.integers(0, N, K)
        t0 = time.perf_counter()
        for r in qs:
            run = int(np.searchsorted(cum, r, "right")) - 1
            _ = book[vals[run]]
        acc_chromo = (time.perf_counter() - t0) / K * 1e6            # microseconds / access

        comp = gzip.compress(raw, 6)
        w = _width(int(ids.max()))
        reps = 40
        t0 = time.perf_counter()
        for r in qs[:reps]:                                          # each cold access = full decompress
            dec = np.frombuffer(gzip.decompress(comp), dtype=f"<u{w}")
            _ = dec[r]
        acc_gzip = (time.perf_counter() - t0) / reps * 1e6

        print(f"{name:<11} {N:>7} {V:>5} | {len(raw):>8} {gz:>8} {xz:>8} "
              f"{chromo:>8} {chromo_gz:>9} | {acc_chromo:>11.2f} us {acc_gzip:>9.1f} us")
    print("-" * 108)
    print("rate: bytes stored (lower=better). access: microseconds per random token fetch (lower=better).")
    print("takeaway: chromosome trades entropy-coding rate for O(log) random access + O(1) neighbour hops;")
    print("gzip/lzma win rate on generic streams but need full decompression for any random access.")


if __name__ == "__main__":
    bench()
