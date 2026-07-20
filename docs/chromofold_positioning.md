# Where ChromoFold fits — placement & advantages (read this first)

One sentence: **ChromoFold is a GPU-resident, random-access, searchable entropy code for the token- and
tensor-shaped data an LLM runs on — not a cold-archive compressor.** It competes with a KV cache / a raw weight
tensor / an index, **not** with gzip or xz. Use it when you need to *touch part of* compressed data on the GPU
without unpacking the whole thing; use xz when you just need the smallest file on disk.

---

## What it is / what it is not

| It **is** | It is **not** |
|---|---|
| an entropy coder that keeps **O(1)/O(log) random access** | a streaming archive coder (gzip/xz/zstd) |
| **GPU-resident** — decode/search in VRAM, no CPU round-trip | a CPU codec you inflate before use |
| **searchable in place** (`count`/`locate`/`predict`) over the compressed form | a black-box blob you must fully decompress to read |
| a **capacity** play: hold more (KV, experts, adapters, context) per GB | a **ratio** play: it does not beat xz on bytes |
| the **terminal** stage of a pipeline (ingests raw values/tokens) | something you stack under/over another compressor |

If a decision needs one rule: **need to read a slice, search, or decode on-GPU → ChromoFold. Need the smallest
cold file and will decompress the whole thing → xz.** They are different tools.

---

## The three advantages, with measured evidence

All numbers from the reproducible records in `docs/` (hardware header in `bench_gpu_results.md`).

1. **Random access without decompressing** (`bench_frontier_results.md`, `bench_stack_results.md`).
   On a 4 M-token stream, serving a sparse read (q ≤ 16 K) is **68–111× faster than a zstd whole-stream
   decompress**; the cost scales with q while decompress-all is fixed. gzip/xz/zstd have **no** random access —
   any single value costs a full-stream inflate.
2. **Everything runs GPU-resident** (`bench_gpu_results.md`). Decode ~**400 M tok/s** (kernel); FM-index search
   `count`/`locate`/`predict` all in VRAM; delta/dedup/weight/KV/MoE reconstruction all on-GPU. The whole point
   is *never leaving the GPU* — the CPU round-trip is exactly what a streaming codec forces.
3. **Search + generation in the compressed domain** (`gpu_fm_index`, `spec_decode.py`). The same bytes are a
   substring index and an n-gram draft model. Demonstrated on a real model: the FM-index used as the **draft in
   speculative decoding** cuts gpt2 forward passes **2.18×** on RAG-flavour text (48→22 passes), output
   byte-identical to greedy — no extra model, no training. No decompressor can do this.

And the ratio gap is now **small**: with the container's monotone index metadata delta-compressed (superblocks
etc., which keeps random access), a real gpt2 int4 weight tensor is **cfold 1.30 b/val vs xz 1.14, gz 1.22** —
within ~13% of xz *while keeping O(1) GPU random access* (`bench_stack_results.md`). ChromoFold is not trying to
win that 13%; it is buying the random access and search that xz can't offer at any ratio.

---

## Decision guide

```
Do you need to read only PART of the data, or search it, or decode it on the GPU?
├─ YES → is it token/tensor-shaped LLM data (context, KV, weights, experts, adapters, dataset)?
│        ├─ YES → ChromoFold  (pick the transform via `autotune.plan` / a preset; see chromofold.md §2–3)
│        └─ NO  → a general succinct index / mmap; ChromoFold if it maps to tokens
└─ NO  → you want the smallest cold file and will inflate the whole thing
         ├─ high-entropy after quant → xz (best ratio; slow)  or  zstd (fast, near-xz)
         └─ archiving a cfold artifact → xz the blob (cold only; this DROPS random access — see below)
```

**Do not chain them in production.** `cfold→xz` shrinks the blob a little more (it re-compresses the same
index metadata) but **destroys random access**, so it's only for a cold tarball — and there xz-direct on the
raw values is smaller and simpler. cfold is *terminal*: a stream compressor before it has nothing to consume,
after it defeats the purpose (`bench_stack_results.md`).

---

## Where it sits per LLM workload (all measured; see `chromofold.md` §4–5)

| workload | alternative it replaces | ChromoFold advantage | measured |
|---|---|---|---|
| prompt cache (shared prefix) | duplicated token cache | 30× store + O(1) span recovery | `prompt_cache.py` |
| mixed prompts (many prefixes) | per-request cache | 21× via N seed chromosomes | `multi_seed.py` |
| LoRA / adapter library | fp16 adapter bank | 30×, byte-identical logits, hot-swap | `lora_library*.py` |
| MoE experts | fp16 expert bank | ~15×, decode only routed | `moe_store.py` |
| KV cache (long context) | fp16 KV | ~5×, attention lossless, attended-only decode | `kv_store.py` |
| dense weights / whole model | fp16 weights | int8 2.7× whole gpt2, generates | `weight_store.py`, `model_store.py` |
| context self-index / RAG | raw tokens + separate index | ~H₀, count+locate+predict in VRAM | `gpu_fm_index.py` |
| dataset dedup | raw shards | 1.74× **with** random access | `dedup.py` |

---

## The honest caveats (so the placement stays true)

- **Not a ratio winner** — xz beats it on bytes everywhere; that's fine, it's not the job.
- **Quantization is the lossy lever**, ChromoFold is the lossless entropy+access layer on top; whole-model
  low-bit quality is quantization's story (a small dense model is embedding-bound at ~3×).
- **Decode has a cost** — random access is cheap per element, but reconstructing a *whole* large tensor is a
  real kernel; the win is capacity + partial reads, not free decompression.
- **Resident vs disk are different footprints** — the "fit more on one GPU" numbers are VRAM-resident; the blob
  ratio (vs xz) is the on-disk story. Both matter; don't conflate them.
