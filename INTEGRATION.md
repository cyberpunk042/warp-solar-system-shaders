# ChromoFold — Integration Guide

How to consume ChromoFold from **Hugging Face** workflows and from a **sovereign / on-prem (air-gapped)**
deployment. The core is offline by design; the model layer is the only part that ever touches a network, and
that is under your control.

## Install

```bash
pip install chromofold            # core: numpy + NVIDIA Warp. No network I/O at import or use.
pip install chromofold[torch]     # + the drop-in transformers KV cache (torch + transformers)
```

`import chromofold` performs no network calls and does not import torch (the transformers cache loads lazily
on first use). Verify:

```python
import chromofold as cf
print(cf.__version__)             # imports nothing heavy, nothing networked
```

---

## Hugging Face

### 1. Drop-in `transformers` KV cache (the headline)

`ChromoFoldCache` is a real `past_key_values`: each layer keeps a small fp16 **residual window** and compresses
the settled prefix (KIVI per-axis quantization + entropy coding), so a model generates normally with the long
context held compressed. Output tracks the quantized-KV model; the resident KV footprint shrinks.

```python
import chromofold as cf
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "Qwen/Qwen2.5-0.5B-Instruct"           # any CausalLM
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name).eval()

inputs = tok(long_prompt, return_tensors="pt")
cache = cf.ChromoFoldCache(residual=128, bits=4, device="cuda:0")
out = model.generate(**inputs, past_key_values=cache, max_new_tokens=64)
print(cf.__version__, "resident KV bytes:", cache.memory_bytes())
```

How it costs: each compressed chunk is decoded **exactly once** (when it settles) and memoized, so per-step
cost is O(new tokens), not O(context) — generation stays O(n), not O(n²). Attention still sees the full K/V, so
you trade a one-time decode per chunk for the memory saving (compute-for-memory). It supports batched and
beam-search generation; `crop` into the compressed prefix (assisted/speculative decoding that rewinds past the
residual window) is not supported — use a larger `residual` or a standard cache for that path. On short contexts
the residual window dominates, so the memory win is smaller; the asymptotic ratio (~8.9× vs fp16) grows with
context.

### 2. Compress weights / token streams

```python
art = cf.compress(weight_tensor)      # 2-D float  -> quantized + entropy-coded, GPU-addressable
art = cf.compress(token_ids)          # 1-D int    -> addressable RRR self-index (search-capable)
art.fetch(indices); art.decode(); blob = art.save()   # portable .cfold container
```

### 3. Publishing to the Hub (optional)

- **Model card / README:** [`chromofold/README.md`](chromofold/README.md) is written to double as a package /
  model-card README.
- **Compressed artifacts:** `Artifact.save()` produces a portable `.cfold` blob you can upload as a Hub file;
  `chromofold`-side `Artifact.load(blob)` rebuilds it on the GPU.
- **Spaces:** the KV-cache long-context demo is a natural interactive Space (compress prefix → generate →
  show resident-KV vs fp16).

---

## Sovereign / on-prem (air-gapped)

ChromoFold is built for data-sovereign deployment: **compute-for-memory on your own hardware, no external
services, no telemetry.**

### Offline guarantee
- The **core** (`chromofold`, `warp_compress`) performs **no network I/O** — no HTTP, no sockets, no
  hub-download at import or during compress / decode / fetch / search. It runs fully air-gapped.
- The **only** network-capable dependency is the model loader (`transformers`), and only when *you* fetch model
  weights. Point it at local weights and pin offline mode:

  ```bash
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1     # never phone home; use local weights only
  ```

- **Air-gap install:** build a wheel where you have connectivity (`python -m build`), vendor it, and
  `pip install --no-index chromofold-0.1.0-py3-none-any.whl` on the sovereign host. Dependencies (`numpy`,
  `warp-lang`, optionally `torch`/`transformers`) are standard wheels you can mirror.

### Data sovereignty
- Compressed data stays **resident on your GPU / host**. Random access, search (FM-index `count`/`locate`),
  and decode all happen in place — nothing is shipped to a service to be inflated.
- **Lossless over the chosen quantization:** the entropy + index layer is bit-exact over whatever quantization
  you apply, so results are reproducible and auditable. Quantization is the only (declared, controllable) lossy
  step.

### Serving-loop integration
- **KV / long context:** use `ChromoFoldCache` as `past_key_values` in your generation loop (above) to fit
  longer context / larger batch in the same VRAM.
- **Weights / MoE / adapters:** hold the bank compressed with the stores and decode on demand:

  ```python
  from chromofold import QuantizedWeightStore, KVCacheStore, MoEExpertStore
  st = QuantizedWeightStore(W, bits=4, group_size=128)   # entropy-coded, GPU-addressable
  st.reconstruct()            # dequantized tensor      st.fetch(idx)   # specific weights, on the GPU
  st.save()                   # portable .cfold blob
  ```

- **Search / retrieval:** compress a corpus of token streams and query it in the compressed domain (FM-index)
  without materializing it — useful for on-prem RAG / prompt-cache span recovery.

### Native path (embedded / linkable)
For deployments that cannot take a Python dependency, a native **C++20 / CUDA C++** engine with a **stable C
ABI** (`libchromofold`, `cf_access_async` / `cf_rank_async` / …) is developed alongside this package and can be
linked directly. The `.cfold` / reference binary formats are the shared interchange. (See the native engine
repository and its `docs/PROJECT_SYNC.md`.)

---

## License

MIT (see `LICENSE.md`). No usage telemetry, no license server, no phone-home.
