# ChromoFold

*A configurable, GPU-resident, random-access compressor for the token-addressable data that language
models live on — contexts, caches, adapters, datasets, and the compressible strata of the weights
themselves. The goal is not "gzip but fancier." The goal is **effective gain**: more useful model on one
GPU, at lower latency, by never leaving the GPU to decompress and never unfolding more than you touch.*

This is the first roll. Everything here is a knob, not a commitment.

---

## 0. Honest preface — metaphor vs. mechanism

The DNA fold you can watch (card → tokens → base pairs → helix → nucleosome → fibre → chromatid → X/Y →
super-chromosome) is a **visualization**. It is *not* the compressor. Do not let the pretty geometry lock us
into anything.

What actually does the compressing is a stack of standard, battle-tested succinct-data-structure pieces:

| ChromoFold stage | what it really is | why it earns its place |
|---|---|---|
| serialize | Hilbert / boustrophedon read order | a locality-preserving 1-D order so neighbours stay near |
| dedup | merge codec (content-addressed blocks) | V≪N distinct pieces; the only "real" data is the id stream |
| transform | **BWT** (→ FM-index) *or* **reference-delta tree** | clusters equal contexts / stores only where things differ |
| code | **RRR / rANS** entropy coding | drives the id stream to its H₀ / Hₖ bound, *keeping O(1) rank* |
| index | **wavelet matrix + sampled suffix array** | random access & substring search *inside* the compressed bytes |

Everything we've already built (`warp_compress/fm_index.py`, `wavelet.py`, `token_chromosome.py`,
`super_chromosome.py`, `entropy.py`, `lm_memory.py`) is exactly this stack. The "chromosome" is just the name
for *one ordered, deduped, entropy-coded, self-indexed token sequence*. A "super-chromosome" is that applied
recursively to a **cluster** of them (reference-delta across near-duplicate sequences). Real mechanism, real
numbers — see `docs/genome_compression_math.md`.

So when this doc says "fold," read: *serialize → dedup → transform → code → index*, with every stage optional
and swappable.

---

## 1. The thesis: effective gain, not compression ratio

The number that matters for serving a model is **not** bytes-saved. It's *useful-work per GB of GPU memory
per unit latency*. Ratio is one term in a sum:

```
effective_gain  =  compression_ratio
                 × (1 − CPU_roundtrip_penalty)      ← gzip pays this; ChromoFold aims for 0
                 × (1 − PCIe_transfer_penalty)      ← you move compressed bytes, not inflated ones
                 × (partial_decode_fraction)⁻¹      ← unfold only the slice you attend to
                 × batch_capacity_multiplier         ← smaller residency ⇒ bigger batches ⇒ more throughput
```

A format that compresses only **1.7×** but **decompresses entirely in GPU memory with random access** can beat
gzip at **2.5×** that requires a CPU inflate + a PCIe copy of the *decompressed* (larger) result. That is the
whole bet. gzip/zstd optimize the first term and ignore the rest; nvCOMP optimizes GPU throughput but is
whole-stream and token-blind. ChromoFold is designed for the **other four terms**:

- **No CPU round-trip.** Decode kernels run on the GPU (roadmap: Warp/CUDA; §7). The compressed bytes live in
  VRAM; you never DMA them to host and back.
- **Partial / random-access unfold.** gzip must inflate from the start of a block. ChromoFold decodes *any
  token, any slice* directly — FM-index positional access is O(1)/O(log); the reference-delta tree reaches any
  token in O(depth)=O(log #sequences). You unfold the 40 KV entries you attend to, not the 4 M you stored.
- **Token-semantic & cross-sequence.** It compresses *ids/tensors*, and it sees redundancy **across**
  sequences (shared prefixes, near-duplicate contexts, sibling adapters) that a byte compressor on one stream
  cannot.
- **Searchable while compressed.** The FM-index answers `count`/`locate`/`predict_next` without
  materialising the sequence — dedup and retrieval happen *in the compressed domain*.

Ratio is the tie-breaker, not the game.

---

## 2. The knobs (this is a pipeline, not a format)

ChromoFold is a **configurable pipeline**. Each workload gets its own point in this space; we find sweet
spots empirically. `warp_compress/chromofold.py` holds the config + presets.

- **serialize**: `hilbert` (best locality, invertible) · `scan` (cheap) · `identity` (already ordered).
- **dedup**: `merge` (content-addressed blocks — huge for shared prefixes/system prompts) · `none`.
- **transform**: `bwt` (searchable self-index, reaches Hₖ) · `delta` (reference + sparse diff, native for
  near-duplicates & LoRA) · `none` (pure positional).
- **code**: `rrr` (succinct, O(1) rank) · `rans` (best ratio, streaming) · `none`.
- **quantize** *(lossy, opt-in)*: `int8`/`int4`/`fp4`/`nf4` — the *big* lever for weights; ChromoFold then
  losslessly squeezes the quantized stream and stores scales/outliers compactly. Compose, don't compete.
- **granularity**: block size / `sa_sample` — the memory⇄latency dial (a sparser suffix-array sample halves
  index memory and doubles locate latency; we measured the curve in `memory_profile.py`).
- **target**: `gpu` (decode in VRAM) · `cpu` (cold storage, ratio-first).
- **random_access**: on ⇒ keep the index; off ⇒ drop it for pure archival ratio.

The point of naming all of these: *nothing is fixed*. If BWT is too hungry for a workload, we run
`delta`+`rans`. If the fold is too slow, we drop to `scan`+`merge`. We tune per stratum.

---

## 3. Where it wins (the workload map)

| workload | why ChromoFold fits | key knobs | status |
|---|---|---|---|
| **prompt caches** | huge shared/system prefixes; dedup + random resume | `merge`+`bwt` | reuse `fm_index` |
| **conversation history** | append-only, near-duplicate turns; delta across turns | `delta`+`rans` | reuse `super_chromosome` |
| **repeated system prompts** | identical across requests; one copy + references | `merge` | reuse `token_chromosome` |
| **batched shared prefixes** | many sequences share a head; store head once | `delta` tree | reuse `super_chromosome` |
| **RAG / retrieval context** | random-access chunks; search-in-place; coarse→fine | `bwt`+hier | reuse `lm_memory` |
| **tokenized datasets** | random minibatch sampling; **near-dup dedup** for training | `bwt`+`merge` | reuse `fm_index` |
| **speculative decoding** | draft buffers; the index itself is a free draft model (`predict_next`) | `bwt` | reuse `generate()` |
| **KV-cache metadata / sparse KV** | masks + indices compress hard; partial unfold of attended entries | `rrr`+`delta` | roadmap |

The common thread: **you rarely need the whole thing at once** — you need a slice, a resume point, a routed
expert, an attended entry. That is precisely what gzip is worst at and ChromoFold is built for.

---

## 4. Fitting a massive model on one GPU — combine with quantization, don't fight it

You said it: not AirLLM. AirLLM streams layers off disk/CPU per forward pass and dies on latency. The opposite
plan is **keep everything GPU-resident but compressed, and unfold just-in-time in VRAM**. Be honest about what
compresses, because dense weights *after* quantization are near-incompressible losslessly. So ChromoFold does
**not** replace quantization — it **stacks on the strata that still have structure**:

- **Dense quantized weights** — quantization is the 4–8× lever (INT4/FP4/NF4). ChromoFold adds **entropy
  coding of the quantized stream** (quantized weights are *not* uniform — their histogram is peaky, so rANS/RRR
  buys another ~5–20%) and stores **scales + outliers** as a sparse side-channel. Net: quant × a bit more,
  losslessly, with GPU decode.
- **MoE experts** — many experts, few active per token. Keep cold experts **ChromoFolded in VRAM**, unfold the
  routed expert on demand. Random access is the entire win; this is where "fit the whole model" gets real,
  because most of an MoE is idle at any step.
- **Sparse / structured-sparse layers** — masks compress enormously (RRR on the bitvector → ~its entropy with
  O(1) rank), values compress on top, and you get O(1) access to the nonzeros.
- **Embedding & LM-head tables** — large, Zipfian, addressed by token id. Positional ChromoFold gives O(1)
  fetch of a row without inflating the table.
- **KV cache** — quantize + delta across time + compress attention-cold entries; unfold only attended
  positions. Grows with context, so this is the strata that most often decides whether a long-context request
  fits.
- **LoRA / adapters / deltas** — low-rank *and* sparse: the reference-delta tree is native. Store a *library*
  of adapters as deltas off the base and hot-swap them for ~free (§6).

The "fit it all on one GPU" claim is the **sum of these**, not magic on dense matmul weights. That's the
honest, buildable version — and it's still a big deal, because the idle strata (cold experts, KV history,
adapter library, embedding tables) are often the majority of the footprint.

---

## 5. Training, LoRA, and better context

**Better context (built).** A long context is stored as a ChromoFold self-index: `V` unique embeddings + the
positional/BWT index + coarse block summaries. The model reads it **coarse→fine** — attend at the
fibre/chromosome scale, drill to tokens on demand (`lm_memory.CompressedContextMemory`, the hierarchical
chromosome). Cost is O(1)/O(log) per access instead of storing & attending over all N. This is an *addressable
compressed KV/context memory*, the same object measured against a 25 GB KV cache in `memory_profile.py`.

**LoRA & adapter libraries (native fit).** LoRA *is* a low-rank delta on the base weights — exactly what the
reference-delta tree stores. So: keep the base once, store each adapter as a ChromoFold delta, and hold a whole
**library** of task adapters compressed in VRAM, swapping the routed one in per request. Sibling adapters
(fine-tunes of a fine-tune) delta against each other, so the library cost is sublinear in adapter count.

**Training-side (roadmap, honest about limits).**
- *Data*: tokenized datasets as ChromoFold shards → random-access minibatch sampling from compressed bytes,
  plus **exact/near dedup** via the FM-index (dedup is a known quality + efficiency win). Solid, near-term.
- *Optimizer state*: Adam m/v are high-entropy floats — limited lossless gain; the realistic play is
  quantize-then-entropy-code the exponents. Modest, not a headline.
- *Speculative decoding*: the FM-index over the running context is a **free n-gram draft model** — we already
  built `generate()` / `predict_next`; use it to propose tokens the big model verifies.

---

## 6. Why it's genius (stated plainly, without the hand-waving)

1. **It attacks the right bottleneck.** LLM serving is bound by memory capacity and bandwidth, not FLOPs.
   A GPU-resident, random-access compressor buys capacity *and* saves bandwidth — the two scarcest resources.
2. **One object, many powers.** The same compressed bytes are *addressable* (O(1) fetch), *searchable*
   (count/locate), *generative* (predict_next), and *a memory* (coarse→fine). No other format in this space is
   all four at once. That's the FM-index's dirty secret, and it's ours.
3. **Partial unfold matches how models actually read.** Attention touches a slice; MoE routes to one expert;
   RAG pulls a chunk; decoding resumes at a point. gzip can't serve a slice; ChromoFold is *built* from slices.
4. **It composes with quantization instead of competing.** The lossy lever (quant) and the lossless lever
   (dedup+entropy+delta) multiply. Most "compression for LLMs" work picks one lane; stacking them is the edge.
5. **It's honest about its metaphor.** The DNA story sells the intuition; the succinct-structure stack does the
   work. We can throw away the geometry and keep every ounce of the compression. Nothing is load-bearing that
   shouldn't be.

---

## 7. Roadmap — first roll → real system

1. **Config + presets** (`warp_compress/chromofold.py`) — one dial-set, per-workload sweet spots. *(this roll)*
2. **GPU decode kernels** — port wavelet `rank`, RRR, and delta-apply to **Warp/CUDA** (we already run Warp for
   rendering). This is what turns "1.7× but GPU-resident" from a claim into a measurement.
3. **Bench the effective-gain terms** — not just ratio: measure PCIe avoided, decode µs on-GPU, batch-capacity
   delta at fixed latency. The equation in §1 becomes a table.
4. **Quantization interop** — wrap INT4/FP4/NF4 as a `quantize` stage; entropy-code the quantized stream;
   sparse scales/outliers side-channel.
5. **MoE + adapter-library demos** — the two clearest "fit more on one GPU" wins; both are random-access
   plays we can prototype on real checkpoints.
6. **Tune everything.** Sweep block size / sa_sample / transform per stratum; publish the curves.

## 8. Honest limitations (so we aim true)

- Dense post-quant weights barely compress losslessly — the win there is small and comes from entropy-coding,
  not folding. Don't oversell it.
- GPU decode kernels don't exist yet; today's code is numpy/CPU. The GPU-resident thesis is *designed for* but
  not *yet measured*. §7.2 is the make-or-break.
- BWT construction is O(n log n) and memory-hungry to *build*; it's cheap to *query*. For write-heavy,
  append-only data prefer the `delta` path.
- rANS/RRR beat gzip on the id stream, but LZ still wins raw ratio on scattered, high-entropy edits — which is
  exactly why ratio is only one term in §1.

---

*ChromoFold's game: fold clusters of tokens into a smaller form and lean on the processor for navigation and
partial unfolding — beat or combine with quantization to fit more capable model on one GPU. First roll. We find
the sweet spots from here.*
