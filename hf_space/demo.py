"""demo — the compute behind the ChromoFold Hugging Face Space (no gradio here, so it stays unit-testable).

Runs CPU-only: builds a ChromoFold KV store on synthetic real-shaped KV, and reports the memory saving,
attention error, and the O(1) windowed-access story. The proven-at-scale curve uses the frozen bench numbers.
"""
from __future__ import annotations

import time

import numpy as np

import chromofold as cf


def _softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def _synth_kv(n, layers, heads, dim, seed=0):
    rng = np.random.default_rng(seed)
    return [((rng.standard_normal((1, heads, n, dim)) * 0.3).astype(np.float32),
             (rng.standard_normal((1, heads, n, dim)) * 0.3).astype(np.float32)) for _ in range(layers)]


def measure(context_len: int, bits: int, layers: int = 4, heads: int = 8, dim: int = 64, window: int = 256,
            device: str = "cpu") -> dict:
    """Build a ChromoFold KV store for `context_len` tokens and report VRAM saving, attention error, and the
    windowed-vs-decode-all access cost. Runs on CPU."""
    context_len = int(context_len)
    window = min(int(window), context_len)
    pkv = _synth_kv(context_len, layers, heads, dim)
    st = cf.KVCacheStore(pkv, bits=int(bits), device=device, per_axis=True)

    fp16_bytes = 2 * layers * heads * context_len * dim * 2
    cf_bytes = st.size_bytes()

    # attention error vs fp32, on a random query (layer 0)
    rng = np.random.default_rng(1)
    Q = (rng.standard_normal((1, heads, 1, dim)) * 0.3).astype(np.float32)
    K0, V0 = np.asarray(pkv[0][0], np.float32), np.asarray(pkv[0][1], np.float32)
    ref = np.einsum("bhqk,bhkd->bhqd", _softmax(np.einsum("bhqd,bhkd->bhqk", Q, K0) / np.sqrt(dim)), V0)
    got = st.attention(0, Q)
    attn_mse = float(np.mean((got - ref) ** 2))

    # windowed (decode only the attended window, O(window)) vs decode-all (O(context))
    pos = rng.integers(0, context_len, window).astype(np.int64)
    st.attention(0, Q, positions=pos, windowed=True)                 # warm
    t0 = time.perf_counter()
    for _ in range(3):
        st.attention(0, Q, positions=pos, windowed=True)
    win_ms = (time.perf_counter() - t0) / 3 * 1e3
    t0 = time.perf_counter()
    st.attention(0, Q, positions=pos, windowed=False)
    all_ms = (time.perf_counter() - t0) * 1e3

    return {
        "context_len": context_len, "bits": int(bits),
        "fp16_mb": fp16_bytes / 1e6, "chromofold_mb": cf_bytes / 1e6,
        "vram_x": fp16_bytes / max(cf_bytes, 1),
        "b_per_val": cf_bytes * 8 / st._vals,
        "attn_mse": attn_mse,
        "window_ms": win_ms, "decode_all_ms": all_ms,
        "speedup_x": all_ms / max(win_ms, 1e-9),
    }


# Proven-at-scale reference (measured on Qwen2.5-1.5B KV, RTX 2080 Ti — see bench_kv_scaling.py).
SCALE_N = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
SCALE_WIN_MS = [1.55, 1.53, 1.61, 1.91, 1.82, 1.62, 1.75]
SCALE_ALL_MS = [2.73, 3.92, 9.47, 13.42, 47.49, 140.03, 464.28]
SCALE_FP16_MB = [1.0, 2.1, 4.2, 8.4, 16.8, 33.6, 67.1]
SCALE_CF_MB = [0.12, 0.24, 0.47, 0.95, 1.89, 3.78, 7.56]


def summary_markdown(m: dict) -> str:
    return (
        f"### Context {m['context_len']:,} tokens · int{m['bits']} KV\n\n"
        f"| metric | value |\n|---|---|\n"
        f"| VRAM (fp16 → ChromoFold) | {m['fp16_mb']:.2f} MB → **{m['chromofold_mb']:.2f} MB** "
        f"(**{m['vram_x']:.1f}× smaller**) |\n"
        f"| attention error vs fp32 | {m['attn_mse']:.2e} |\n"
        f"| decode-all vs windowed access | {m['decode_all_ms']:.2f} ms → **{m['window_ms']:.2f} ms** "
        f"({m['speedup_x']:.1f}× on a {256}-token window) |\n\n"
        f"*ChromoFold keeps the KV compressed **and** randomly addressable: a window fetch decodes only what it "
        f"reads (O(window)), so cost does not grow with context length. Lossless over the int{m['bits']} "
        f"quantization; runs on CPU here.*"
    )
