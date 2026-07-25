# Research 47 — three new compression levers for ChromoFold

> Studying `warp_compress/` (the real codec) and the C1/C2/C3 + super-chromosome "versions" (Research
> [44](44-warp-compression.md), [45](45-simulation-and-compression.md)) surfaced one clear opening: the
> project's edge is **addressability + search, not raw ratio** (on ratio, zstd/xz win everywhere and the docs
> say so). So the genuinely-new levers are the ones that either add the *missing lossy lever without breaking
> random access*, or push the *addressable-memory* thesis the docs flag but never ran. This note adds three,
> each a real module + test + measured numbers.

## The gap this closes

Before this round, ChromoFold's only lossy lever was **scalar integer quantization** (`weight_store`) plus a
scalar-outlier side-channel. The whole space of vector/codebook quantization, fixed-distortion semantic
merging, and using the addressable index as an LLM *memory* was unbuilt — even though the succinct layer
(`gpu_wavelet` → `gpu_rrr` → FM-index) that would carry it already exists and is measured.

**Honesty note on measurement.** These were built in a CPU-only container with **no torch and no model access**
(the PyTorch and Hugging Face hosts are policy-blocked here). So the numbers below are **rate–distortion on
realistic synthetic tensors** — the same style as `weight_store`'s own `_demo`. The **live-model** evaluations
(perplexity vs bits/weight; attention quality vs the full KV cache) are the flagged follow-ups for a box with
the model. Every codec is **lossless over its own lossy lever** and **randomly addressable** — both fully
verified here.

---

## Lever 1 — product/vector quantization (`vq_store.py`)

Scalar int-quant is a per-*element* codebook of evenly-spaced points. Real weight rows correlate, so a
**vector codebook** does better per bit: split each row into `subdim`-length sub-vectors, k-means to
`2**codebook_bits` centroids, store the code ids. **A PQ code is just a token**, so the code stream drops
into the same `RRRWaveletGPU` self-index — weights stay GPU-addressable; the lossy lever is the centroid
assignment, the layer on top is lossless over the fp16 codebook (`fetch == reconstruct`).

Measured (`python -m warp_compress.vq_store`, synthetic correlated weights, 2048×512):

| config | b/weight | MSE vs fp32 | output error |
|---|---|---|---|
| int4 per-tensor | 2.92 | 2.04e-4 | 1.04e-1 |
| int3 per-tensor | 1.66 | 1.11e-3 | 5.62e-1 |
| int2 per-tensor | 0.27 | 3.87e-3 | 1.98e+0 |
| **PQ subdim4 6b** | **1.69** | **7.48e-4** | **3.78e-1** |
| **PQ subdim8 8b** | **1.16** | 1.27e-3 | 6.48e-1 |

**Where it wins:** the sub-2-bit regime. At ~1.7 b/weight PQ beats int3 on **both** MSE and output error; at
~1.2 b/weight it holds usable quality where scalar int2 collapses. **Honest negative:** at int4+ (≥~2.9
b/weight) scalar-int + entropy is competitive — PQ is the low-bit lever, not a universal replacement. Balanced
k-means makes the codes near-uniform, so the RRR index here buys *addressability*, not extra ratio (the ratio
is the vector codebook, `log2(k)/subdim`).

---

## Lever 2 — the within-tolerance semantic merge (`semantic_merge.py`)

Every merge in ChromoFold so far is **byte-exact** (`np.unique`). Research 44/45 flag a *semantic* tier —
"merge near-synonym blocks that agree **within tolerance**" — as the one genuinely-new lever. This is it:
**leader clustering** with an L2 tolerance `tol`. It is the complement of PQ — PQ is fixed-*rate*, this is
fixed-*distortion* with a **hard per-row guarantee** (every reconstructed row within `tol`). Assignment ids go
in the same index; unlike PQ codes they are **skewed**, so entropy coding earns real ratio.

Headline application — the **KV cache** (neighbouring tokens' K/V are near-duplicate). Measured
(`python -m warp_compress.semantic_merge`, synthetic near-synonym KV, 4096 tokens × 64):

| tol | leaders | ratio | b/value | max‖Δrow‖ | attn out-err |
|---|---|---|---|---|---|
| 0.00 | 4096 | 1.00 | 16.13 | 0.003 | ~0 (exact) |
| 0.70 | 572 | 7.16 | 2.33 | 0.700 | 4.9e-5 |
| 0.90 | 133 | 30.80 | 0.58 | 0.891 | 1.9e-4 |

A tunable, **bounded-error** lossy tier: `tol=0` is exact byte-merge; raising it merges more near-duplicates
(ratio ↑, bits ↓) for a graceful, measured rise in attention-output error that stays tiny (**31× KV at ~1.9e-4
output MSE** here). The `max‖Δrow‖` column confirms the guarantee (≤ tol). Real-model attention quality is the
follow-up.

---

## Lever 3 — the addressable, searchable context memory (`context_memory.py`)

The docs' explicit unrun experiment: hold a long context as a chromosome and read it at O(1), **searchable in
the compressed domain** — what a flat KV cache is not. This wires it from the pieces already here and **ties
the three levers together**: `leader_merge` (Lever 2) bridges continuous embeddings → a concept book + ids;
`token_chromosome.Chromosome` gives O(1) `at()` + an RLE id stream; `fm_index.FMIndex` makes the id stream
searchable (`count`/`locate`).

Measured (`python -m warp_compress.context_memory`, synthetic concept context, 8192 tokens × 64):

| tol | V (book) | compression | max‖Δrow‖ | `at()`==decode |
|---|---|---|---|---|
| 0.00 | 8192 | 0.5× | 0.003 | ✓ |
| 0.70 | 1058 | 3.6× | 0.700 | ✓ |
| 0.90 | 242 | 16.5× | 0.898 | ✓ |

- **Search** (compressed domain): a concept 3-gram's FM occurrences == brute force, exactly.
- **Retrieval**: 100% of 500 noisy queries map to a concept within `tol` of the true token.
- **Honest:** at `tol=0` an exact book **expands** a fully-distinct context (0.5×) — you still gain search +
  addressing; the compression comes from concept redundancy at `tol>0`, at a bounded per-row error.

The remaining piece — does navigating this memory at inference match full-context quality at lower memory? — is
the model-gated experiment.

---

## How the three compose

```
embeddings / KV / weights
        │
   ┌────┴─────────────────────────────────────────────┐
   │ lossy lever (pick one, bounded/known distortion)  │
   │   • vq_store          fixed-RATE  (bit budget)    │
   │   • semantic_merge    fixed-DISTORTION (tol floor)│
   └────┬─────────────────────────────────────────────┘
        │  ids / codes  (tokens)
        ▼
   RRRWaveletGPU self-index  →  randomly addressable on the GPU  (+ FM-index → searchable)
        │
        ▼
   context_memory:  a long context as an addressable + searchable + coarse→fine chromosome
```

Each lossy lever emits *tokens*; the existing lossless GPU-addressable entropy+index layer carries them
unchanged. The levers are choices along one axis (bit-budget vs quality-floor vs exact); the memory is the
application that consumes an addressable+searchable id stream.

## What's verified here vs. the follow-ups

| Verified in this container (CPU, synthetic) | Follow-up (needs the model box) |
|---|---|
| Rate–distortion curves (bits vs MSE / output error / attn error) | Perplexity vs bits/weight on a real model (PQ) |
| Lossless over the lossy lever; `fetch == reconstruct` | Attention quality vs the full KV cache (semantic merge) |
| GPU-addressable codes; FM search == brute force | LM utility of navigating the context memory vs full context |
| Per-row distortion bound (semantic tier) | Fused decode-GEMM from a PQ codebook (a Marlin-class kernel) |

## Cross-references

- [Research 44 — warp compression](44-warp-compression.md) · [Research 45 — simulation & compression](45-simulation-and-compression.md)
- Code: `warp_compress/{vq_store,semantic_merge,context_memory}.py` · tests `tests/test_{vq_store,semantic_merge,context_memory}.py`
- Reused seams: `weight_store.py`, `gpu_rrr_wavelet.py`, `token_chromosome.py`, `fm_index.py`
