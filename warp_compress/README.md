# ChromoFold

**A GPU-resident, random-access, *searchable* entropy code for the data an LLM runs on** — weights, KV cache,
MoE experts, LoRA adapters, prompt caches, context, and token streams. It competes with a KV cache / raw
weights / an index, **not** with gzip or xz: cfold spends the GPU's cheap compute on succinct-structure math so
compressed data stays *navigable in place* — addressable, searchable, decodable — instead of an opaque blob you
must inflate. Compute-for-memory, aimed where a GPU is bottlenecked (VRAM + bandwidth).

## One API

```python
from warp_compress.api import compress, Artifact

art = compress(W)                 # a 2-D float weight tensor -> quantized + entropy-coded, GPU-addressable
art = compress(tokens)            # a 1-D int token stream    -> addressable RRR self-index
art = compress([s1, s2, ...])     # a batch of sequences      -> seed / delta / dedup (auto-detected)

art.decode()          # reconstruct the whole thing
art.fetch(idx)        # O(1)/O(log) GPU random access (weights / tokens)
art.size_bytes()      # compressed footprint
blob = art.save()     # a portable, versioned .cfold container  (weights & tokens)
Artifact.load(blob)   # rebuild it on the GPU
```

CLI: `python -m warp_compress.cli inspect file.cfold | demo | modules`.

## What it buys (all measured — see `docs/*_results.md`)

| capability | number |
|---|---|
| decode / rank / access, GPU-resident | ~400 M tok/s, ~1.1 B/tok index (≈ H₀; BWT ≈ Hₖ) |
| search in the compressed domain | `count` / `locate` / `predict_next` in VRAM |
| build the index on the GPU too | suffix array by prefix-doubling, **17–32×** vs CPU, bit-identical |
| random access at scale vs decompress-all | **68–111×** faster (sparse reads, 4 M-token stream) |
| speculative decoding (index as draft) | **2.18×** fewer gpt2 forward passes, output = greedy |
| weights (int4 + class-Huffman) | ~1.33 b/w, ~12× vs fp16, lossless over quant |
| weights + SpQR outlier side-channel | int4+1% outliers *beats int8's accuracy at int8's size* |
| MoE expert bank | ~15× (hold all, decode only routed) |
| KV cache (long context) | ~5×, attention lossless, attended-only decode |
| LoRA / adapter library | 30×, byte-identical logits |
| prompt cache (shared prefix) | 30×; mixed prompts 21× (N seed chromosomes) |
| whole gpt2, compressed, generating | int8 → 92.9 MB `.cfold` (vs fp16 249 MB) |
| int4 made *usable* on gpt2 (measured) | **group-128 int4 PPL 30.86 vs fp32 26.62** (per-tensor int4 breaks) |
| **lossless** bf16 (exponent coded) | 1.41× (~11.3 b/val), exact bits, random access |
| decode-in-the-matmul (fused) | weights decoded in-GEMM, **10.6× less VRAM** held vs dense W |

**Not a ratio play:** xz beats cfold on bytes everywhere (that's fine — different job). With cfold's own index
metadata delta-compressed, a gpt2 int4 tensor is ~1.30 b/val vs xz 1.14 — within ~13% *while keeping O(1) GPU
random access* xz can't offer at any ratio.

## Read next

- **Placement & advantages:** [`docs/chromofold_positioning.md`](../docs/chromofold_positioning.md) — when to
  use cfold vs xz, a decision guide, the three advantages with evidence.
- **The engine & knobs:** [`docs/chromofold.md`](../docs/chromofold.md) — the pipeline (serialize → dedup →
  transform → entropy-code → index), presets + auto-detect + manual config.
- **The format & protocol:** [`docs/chromofold_format.md`](../docs/chromofold_format.md) — the `.cfold`
  container byte layout, section registries, encode/decode algorithm, versioning.
- **Lineage & thesis:** [`docs/chromofold_related_work.md`](../docs/chromofold_related_work.md) — the honest
  prior art (nvCOMP, Deep Compression / DFloat11, FM-index, infini-gram, GPTQ/AWQ, KVQuant) and what's
  distinctive (the unification across strata).
- **The math:** [`docs/genome_compression_math.md`](../docs/genome_compression_math.md).

*v1: the individual techniques are known (this is a virtue — the direction is validated); the contribution is
the GPU-resident, random-access, searchable **unification** across every LLM stratum, with a documented format.
There is a lot of borrowable headroom.*
