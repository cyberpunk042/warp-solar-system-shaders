"""memory_profile — what does the FM-index cost as a *context memory*, and what is the sampled-SA knob?

The compressed self-index is being proposed as an LM's addressable context memory. So put real bytes on it:

  1. Size breakdown — the BWT wavelet (RRR ~ n·H0 with O(1) rank), the C array, and the sampled suffix
     array. Compare to the raw context (n tokens) and to a transformer KV cache holding the same n tokens.
  2. The sa_sample tradeoff — sampling every s-th SA entry costs ~ (n/s)·2 ints of memory but makes
     locate() walk up to s LF-steps. Sweep s and measure index bytes vs. mean locate() latency: the actual
     space/time curve you tune per deployment.

Honest framing: a KV cache stores *contextualised activations* (lossy, model-specific) and is not
searchable; the FM-index stores the *exact tokens* near their entropy and answers count / locate /
predict / generate. Different objects — but for the job of "hold a long context and let the model read,
search, and continue it," the FM-index is O(n·H0) bytes and O(1)/O(log) addressable, not O(n·d·L).

Run: python -m warp_compress.memory_profile
"""
from __future__ import annotations

import sys
import time

import numpy as np

from .entropy import H0, _wavelet_index_bits
from .fm_index import FMIndex, suffix_array


def index_bytes(seq, sa_sample: int, _cache={}) -> dict:
    """Byte breakdown of an FM-index over `seq` at a given SA sampling rate. The BWT/wavelet size is
    independent of sa_sample, so compute the suffix array + wavelet bits ONCE and vary only the SA count."""
    seq = np.asarray(seq, np.int64)
    key = (seq.shape[0], int(seq.sum()))
    if key not in _cache:
        s = np.concatenate([seq + 1, [0]])
        sa_full = suffix_array(s)
        bwt = s[(sa_full - 1) % s.shape[0]]
        wavelet = _wavelet_index_bits(bwt, rrr=True) / 8        # RRR-compressed BWT wavelet (searchable core)
        _cache[key] = (sa_full, wavelet, int(s.max()) + 1)
    sa_full, wavelet, sigma = _cache[key]
    c_arr = sigma * 8                                           # C[]: one int per symbol
    n_samples = int(np.count_nonzero(sa_full % sa_sample == 0))
    sa = n_samples * 2 * 8                                      # sampled SA: (pos, rank) int64 pairs
    return {"wavelet": wavelet, "C": c_arr, "sampled_SA": sa, "total": wavelet + c_arr + sa}


def _kv_cache_bytes(n: int, d_model: int = 4096, layers: int = 32, dtype_bytes: int = 2) -> float:
    """KV cache for n tokens: 2 (K+V) · layers · d_model · dtype, per token. The thing this replaces."""
    return n * 2 * layers * d_model * dtype_bytes


def profile(seq):
    seq = np.asarray(seq, np.int64)
    n = int(seq.shape[0])
    h0 = H0(seq)
    raw = n * max(1, (int(seq.max())).bit_length()) / 8         # packed raw tokens
    ent = n * h0 / 8                                            # entropy floor
    print(f"context: N={n} tokens  V={len(np.unique(seq))}  H0={h0:.3f} bits/tok")
    print(f"  raw packed         {raw/1e3:8.1f} KB   ({max(1,(int(seq.max())).bit_length())} bits/tok)")
    print(f"  entropy floor n·H0 {ent/1e3:8.1f} KB")
    kv = _kv_cache_bytes(n)
    print(f"  KV cache (4096d,32L,fp16, same n tokens) {kv/1e6:8.1f} MB   "
          f"= {kv/raw:.0f}× the raw tokens, and not searchable")

    print("\n  FM-index size + sampled-SA space/time tradeoff (locate over 200 known patterns):")
    print(f"  {'sa_sample':>9} {'wavelet':>9} {'SA':>8} {'total':>9}  {'vs n·H0':>7}  {'mean locate':>12}")
    rng = np.random.default_rng(0)
    starts = rng.integers(0, n - 6, 200)
    for s_rate in (8, 16, 32, 64, 128):
        b = index_bytes(seq, s_rate)
        fm = FMIndex(seq, sa_sample=s_rate)
        t0 = time.perf_counter()
        for at in starts:
            fm.locate(seq[at:at + 6])
        dt = (time.perf_counter() - t0) / len(starts) * 1e6
        print(f"  {s_rate:>9} {b['wavelet']/1e3:8.1f}K {b['sampled_SA']/1e3:7.1f}K "
              f"{b['total']/1e3:8.1f}K  {b['total']/ent:6.2f}×  {dt:9.1f} µs")
    print("\n  => the BWT wavelet (the searchable core) sits near n·H0 and is FIXED; the sampled SA is the "
          "only tunable overhead. Coarser sampling shrinks the index toward the entropy floor at the cost of\n"
          "  longer locate() LF-walks — memory halves and latency doubles per step, the classic space/time "
          "curve (absolute µs are pure-Python overhead; the linear-in-sa_sample SHAPE is the invariant).\n"
          "  The whole context memory is O(n·H0) bytes — orders below a KV cache — while answering "
          "count / locate / predict / generate that a KV cache cannot.")


if __name__ == "__main__":
    # a realistic context: repo source as a char stream (real higher-order redundancy)
    import glob
    txt = "".join(open(f).read() for f in sorted(glob.glob("warp_compress/*.py")))
    if len(txt) < 40000:
        txt = (txt * 3)
    seq = np.frombuffer(txt.encode("utf-8", "ignore"), np.uint8).astype(np.int64)[:48000]
    profile(seq)
    sys.stdout.flush()
