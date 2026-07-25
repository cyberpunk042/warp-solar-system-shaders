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

## Lever 4 — low-rank factorization as a codec (`lowrank_store.py`)

Low-rank was absent as a *storage* codec — it only appeared via LoRA *deltas* — though attention/MLP weights
carry real low-rank structure. This factors W ≈ A·B (truncated SVD, fp16 thin factors) and hands the residual
R = W − A·B to an *element* codec (`vq_store` PQ or `weight_store` int). A **two-stage codec**: low-rank
captures the correlated structure, the residual mops up the rest; rows decode addressably (`A[row]·B +
residual[row]`).

Measured (`python -m warp_compress.lowrank_store`, synthetic rank-16 + full-rank noise, 2048×512):

| config | b/weight | MSE vs fp32 | output error |
|---|---|---|---|
| int4 per-tensor | 2.92 | 2.04e-4 | 1.04e-1 |
| low-rank r16 only | **0.62** | 1.88e-3 | 9.57e-1 |
| low-rank r16 + PQ residual | 2.88 | 1.82e-4 | 9.25e-2 |
| low-rank r16 + int4 residual | 3.70 | 7.67e-5 | 3.89e-2 |

**Honest read** (5-seed sweep): low-rank *alone* is an ultra-cheap 0.62 b/weight but drops the noise, so its
output error is **worse** than int4 — structure-only is an extreme operating point, not a quality win.
Composed at matched bits (~2.9), low-rank + PQ-residual is **competitive** with int4 (roughly tied, usually a
touch better on output error) — not a blanket win. Its distinctive value is separating the tensor into an
**addressable low-rank base** (shareable across a model family — the LoRA insight generalized) + a residual,
and owning the **sub-1-bit structure-only** regime.

## Lever 5 — residual (multi-codebook) vector quantization (`rvq_store.py`)

Extends Lever 1: instead of one codebook per sub-vector, stack `stages` of them — codebook 1 quantizes the
sub-vector, codebook 2 quantizes the *residual*, and so on; reconstruction is the sum of the chosen centroids.
Each stage's codes are their own token stream in the same index (addressable), lossless over the fp16
codebooks.

Measured (`python -m warp_compress.rvq_store`, synthetic correlated weights, 2048×512):

| config | b/weight | MSE vs fp32 | output error |
|---|---|---|---|
| PQ subdim4 8b (1 codebook) | 2.25 | 3.92e-4 | 1.98e-1 |
| RVQ subdim4 2×4b | 2.27 | 4.90e-4 | 2.49e-1 |
| RVQ subdim4 3×4b | 3.39 | 1.88e-4 | 9.63e-2 |
| RVQ subdim4 4×4b | 4.45 | 7.73e-5 | 3.94e-2 |

**Honest read:** at *matched* code-bits a single free codebook is slightly **better** than stacked ones (RVQ
2×4b MSE 4.9e-4 vs a single 8b's 3.9e-4) — additive centroids (a Minkowski sum of 16+16 points) are less
expressive than 256 free points. RVQ is **not** a same-bits distortion win. Its value is **scalability + tiny
stable codebooks**: adding stages refines the residual to high accuracy (4×4b → 7.7e-5 MSE) using 4×16
centroids, where the equivalent single codebook needs 2¹⁶ = 65536 (infeasible k-means + huge codebook). So it
is the "scale PQ to high accuracy with small codebooks" lever — useful for per-tensor codebooks and small
tensors where a big codebook doesn't amortize.

## Lever 6 — 2:4 structured sparsity (`sparse_store.py`) — a measured *negative*

The one lever category left untouched was structured sparsity. NVIDIA Ampere+ tensor cores run a **2:4**
pattern (exactly 2 of every 4 contiguous weights nonzero) at ~2× throughput. Implemented as a codec:
magnitude-prune each group of 4 to its two largest, store the survivors (int, entropy-coded) + a 3-bit pattern
id per group; both streams in the RRR index (addressable).

Measured (`python -m warp_compress.sparse_store`, Gaussian weights, **no fine-tuning**):

| config | b/weight | MSE | output error |
|---|---|---|---|
| dense int4 | 3.05 | 8.30e-5 | 4.22e-2 |
| dense PQ 4×8 | 2.26 | 1.89e-4 | 9.60e-2 |
| 2:4 int8 | 4.77 | 2.59e-4 | **1.30e-1** |
| 2:4 int4 | 2.46 | 3.00e-4 | 1.51e-1 |

**Honest negative:** post-training 2:4 drops half the weights, so it is **dominated** on the RD curve — 2:4-int8
spends 4.77 b/w yet has **3× the output error** of dense int4 at 3.05 b/w. 2:4 is *not* a ratio/quality win
here; its payoff is the ~2× sparse-tensor-core **throughput** on supported GPUs (not measurable on CPU), which
you pair with fine-tuning to recover the accuracy this number quantifies. Shipping it as a measured negative is
the point — the lever is mapped, its cost known. (The lever selector deliberately does *not* include it, since
it loses on the pure RD metric it optimizes.)

## The capstone — a build-driven lever selector (`lever_select.py`)

Five weight levers (int, PQ, RVQ, low-rank+residual) are choices with different sweet spots. `lever_select`
turns them into **one entry point** that, like `autotune` elsewhere in ChromoFold, **builds a candidate per
lever and measures it** (bits, MSE, output error), then picks the best for the goal — minimise output error,
optionally under a bit budget — and reports the SVD structure signal that explains the pick. Build-driven, so
it can never mis-label.

Measured (`python -m warp_compress.lever_select`, three tensor archetypes, budget sweep):

| tensor (top-8 SV energy) | @2.5 b/w | @3.6 b/w | @4.5 b/w |
|---|---|---|---|
| low-rank r8 (**53%**) | PQ 4×8 | **lowrank16+PQ** | int4+outliers |
| full-rank Gaussian (7%) | PQ 4×8 | **int4** | int4+outliers |
| heavy-tailed outliers (8%) | PQ 4×8 | lowrank16+PQ | **int4+outliers** |

The pick genuinely varies with the tensor and the budget: the low-rank tensor (53% of its energy in the top-8
singular values) picks the **low-rank lever** at a mid budget; the full-rank tensor (7%) picks **int4** because
low-rank offers it nothing; the outlier tensor picks the **outlier side-channel** when the budget allows. The
structure signal in each `reason` string is exactly what explains the choice — no guessing, all receipts.

## How the levers compose

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
