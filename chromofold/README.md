# ChromoFold

**GPU-resident, random-access, *searchable* compression for the data an LLM runs on** — KV cache, weights,
MoE experts, adapters, prompt caches, and token streams. ChromoFold keeps that data compressed **and navigable
in VRAM**: addressable in O(1)/O(log) and searchable in the compressed domain, decoded only where it is
consumed. It composes **on top of** quantization — it is the lossless entropy + random-access layer, not the
quantizer.

It is not a `gzip`/`zstd` competitor (those win on bytes and lose navigation). It competes with a raw KV cache,
raw weights, or a separate index: it spends the GPU's cheap compute to buy back scarce **VRAM and bandwidth**.

## Install

```bash
pip install chromofold            # core (numpy + NVIDIA Warp) — offline, no network at import
pip install chromofold[torch]     # + the drop-in Hugging Face transformers KV cache
```

## Quick start

```python
import chromofold as cf

# compress a weight tensor or a token stream -> a GPU-addressable Artifact
art = cf.compress(weight_tensor)          # 2-D float -> quantized + entropy-coded
art.fetch(indices)                        # O(1)/O(log) random access, on the GPU
art.decode(); art.size_bytes(); art.save()

# drop-in KV cache for transformers (compress the settled prefix, keep a small fp16 window)
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
cache = cf.ChromoFoldCache(residual=128, bits=4)
out = model.generate(**inputs, past_key_values=cache)   # normal generation, compressed KV
cache.memory_bytes()                                     # resident KV footprint
```

## What it buys (measured)

- **KV cache, long context** — window-attention latency stays *flat* from 1K→64K tokens while decompress-all
  grows (264× at 64K on Qwen2.5-1.5B), at ~8.9× less VRAM than fp16; attention error 1.5e-4 (KIVI int4).
- **Random access** — 68–111× faster than decompress-all for sparse reads.
- **Weights** — int4 + class-Huffman ~1.33 b/w; SpQR-style outliers make int4 beat int8's accuracy at int8's
  size; whole models still generate (int8 near-lossless).
- **Search in the compressed domain** — FM-index `count`/`locate`/`predict_next` in VRAM (n-gram draft model).

## CLI

```bash
chromofold info        # version, offline guarantee, GPU availability
chromofold selftest    # compress -> reconstruct round-trip — run this to validate an install / air-gap deploy
chromofold inspect model.cfold
```

Runnable examples in [`examples/`](../examples/): `compress_weights.py` (offline, no torch) and
`serve_compressed_kv.py` (long-context generation through the compressed KV cache).

## Honest scope

Quantization is the lossy lever; ChromoFold's entropy + index layer is **lossless over the chosen quantization**
and randomly addressable. It does not beat `xz` on raw ratio (different job). The reference implementation is
Python + NVIDIA Warp; a native C++/CUDA engine with a stable C ABI is developed alongside it.

See **[INTEGRATION.md](../INTEGRATION.md)** for Hugging Face and sovereign / on-prem (air-gapped) deployment.

MIT licensed.
