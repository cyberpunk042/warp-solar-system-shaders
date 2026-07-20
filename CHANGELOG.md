# Changelog

All notable changes to the `chromofold` package.

## 0.1.0 — first packaged release

The reference implementation (Python + NVIDIA Warp) packaged for Hugging Face and sovereign / on-prem use.

**Package**
- Installable `chromofold` distribution (`pip install chromofold` / `chromofold[torch]`), MIT.
- Clean public API via a lazy shim over `warp_compress`: `import chromofold` does **no network I/O** and does
  **not** import torch (the transformers KV cache loads on first use). Verified installable in an isolated,
  dependency-free environment.
- `chromofold` CLI: `info`, `selftest` (compress→reconstruct round-trip to validate an install / air-gap
  deploy), `inspect FILE.cfold`.
- Examples: `examples/compress_weights.py`, `examples/serve_compressed_kv.py`.
- Guides: `INTEGRATION.md` (Hugging Face + sovereign/on-prem, the no-phone-home guarantee).

**Public API**
- `compress(data) -> Artifact` (weights / tokens); `Artifact` (`.decode`/`.fetch`/`.save`/`.load`/`.size_bytes`).
- `QuantizedWeightStore`, `KVCacheStore`, `MoEExpertStore`.
- `ChromoFoldCache(residual=, bits=)` — drop-in transformers `past_key_values`.

**ChromoFoldCache (drop-in transformers KV cache)**
- Compresses the settled prefix (KIVI per-axis + entropy), keeps a small fp16 residual window.
- **O(n) generation, not O(n²):** each compressed chunk is decoded exactly once (memoized), not re-decoded per
  step. Output bit-identical to the decode-all path.
- Supports batched and beam-search (`reorder_cache`) generation. `crop` into the compressed prefix is refused
  with a clear error (a larger `residual` or a standard cache handles that path).
