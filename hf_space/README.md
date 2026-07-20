---
title: ChromoFold Compressed KV Cache
emoji: 🧬
colorFrom: orange
colorTo: gray
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
license: mit
short_description: A compressed, randomly-addressable KV cache that stays searchable — measured on CPU.
---

# ChromoFold — Hugging Face Space

An interactive demo of **ChromoFold**: a GPU-resident, random-access, *searchable* compression layer for LLM
data. This Space shows the KV-cache story on **CPU** (no GPU, no model download): pick a context length and KV
bit-width, and it builds a real ChromoFold KV store and reports the memory saving, attention error, and the
windowed-vs-decode-all access cost. The plots show the same behavior **measured on a real model
(Qwen2.5-1.5B)** — windowed access stays flat to 64K tokens while decompress-all explodes.

## Files
- `app.py` — the Gradio UI.
- `demo.py` — the compute (gradio-free, unit-testable); builds a `chromofold.KVCacheStore` and measures.
- `requirements.txt` — dependencies.

## Deploying
1. Create a **Gradio** Space on Hugging Face.
2. Add `app.py`, `demo.py`, and `requirements.txt`.
3. Make `chromofold` importable — until it is published to PyPI, either add it to `requirements.txt` as a
   `git+` install, or copy the `chromofold/` and `warp_compress/` packages into the Space repo.
4. CPU hardware is sufficient (the demo runs the compression on CPU).

Full project, docs, and the native C++/CUDA engine:
<https://github.com/cyberpunk042/warp-solar-system-shaders>. Honest scope: quantization is the lossy lever;
ChromoFold's entropy + index layer is lossless over the chosen quantization and randomly addressable.
