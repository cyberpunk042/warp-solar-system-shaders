# What ChromoFold is, and who came before — the honest lineage

*This is the intellectually-honest companion to the positioning doc. ChromoFold (v1) reuses a lot of known
ideas; being clear about that is a feature — it means the direction is validated, and it points at where the
real room for improvement is. **Citations below were web-verified (July 2026)** with arXiv IDs and reported
numbers; the improvement leads each name a concrete method to borrow.*

---

## The thesis in one line

**ChromoFold spends the GPU's cheap compute on succinct-structure math so that compressed LLM data stays
directly navigable — addressable, searchable, decodable *in place* — instead of being an opaque blob you must
inflate.** It is a **compute-for-memory** trade, aimed at exactly the resource that is scarce on a GPU: VRAM
capacity and memory bandwidth (FLOPs are comparatively abundant).

The distinguishing constraint vs. ordinary compression: *ordinary compression also does math to shrink data,
but it destroys navigation.* ChromoFold's bet is to keep **O(1)/O(log) random access and search in the
compressed domain** — it is **self-indexing**, not merely compressed. That is the whole reason it belongs on a
GPU and next to a KV cache, not on a disk next to a tarball.

---

## Prior art — every ingredient has a lineage (web-verified, July 2026)

| ChromoFold piece | Prior art (verified) | How ours differs / overlaps |
|---|---|---|
| GPU-resident (de)compression | **nvCOMP** (GDeflate 32-way parallel; **gANS** ~7× throughput; bitcomp, cascaded) | nvCOMP is streaming/block; ours adds per-element **random access + search**, token/tensor-aware |
| entropy-coding weights + GPU decode | **DFloat11** (arXiv 2504.11651, NeurIPS'25): Huffman-code the **BF16 exponent** (~2.6 bits real info), ~70% size / "11-bit", 100% acc, SRAM LUT decode, **20.97× faster decode than nvCOMP**. **NeuZip** (arXiv 2410.20650, NeurIPS'24): **ANS** on exponents; Llama-3-8B train 31→<16 GB. **Deep Compression** (Han 2016) | **DFloat11 is the closest — and confirms our distinctiveness: it decodes WHOLE tensors (no random access), WEIGHTS-only, and is lossless-on-BF16 (not on quantized ints).** Ours is random-access, all-strata, over quantized values |
| the lossy quantizer we compose with | **GPTQ**, **AWQ**, **QuIP#**, **bitsandbytes NF4/QLoRA**, **SpQR** (sparse outliers), **XFP** (codebook + outlier separation) | we don't quantize better; we entropy-code + address the quantized stream losslessly |
| KV-cache shrink | **KIVI** (ICML'24, arXiv 2402.02750): **per-channel Keys, per-token Values**, 2-bit, 2.6× mem, 2.35–3.47× throughput. **KVQuant** (Hooper 2024): per-channel-K outlier-aware, 10M ctx. **RotateKV** (2501.16383). **H2O**; vLLM PagedAttention (mgmt) | **none of them entropy-code the quantized KV — they quantize/evict only.** Ours adds the lossless entropy layer + attended-only decode (should adopt KIVI's per-channel-K/per-token-V) |
| the self-index (rank/select) | **FM-index** (Ferragina–Manzini), **wavelet trees** (Claude–Navarro), **RRR** | textbook succinct structures, on the GPU over token streams |
| GPU FM-index / BWT · device-resident RA decode | genomics aligners **nvBIO / SOAP3-dp / CUSHAW**; **2026 "Compressed-Resident Genomics"** (arXiv 2606.18900, 2606.24531): device-resident GPU LZ77+entropy decode with **position-invariant random access** | closest to our random-access-in-VRAM claim — but genomics (LZ77), not LLM entropy-coded quantized data |
| n-gram LM / retrieval speculative decoding | **infini-gram** (Liu 2024); **Prompt-Lookup Decoding**; **REST** (retrieval + datastore); n-gram drafts ~1.1–2.9× | we rebuilt the suffix-index LM independently; our `spec_decode` (2.18×) is the PLD/REST family, drafting from the *compressed self-index* |
| entropy coder | **Huffman**; **rANS/ANS** (Duda 2013; zstd-FSE, nvCOMP gANS, NeuZip) | v1 uses RRR + canonical Huffman; **rANS is the v2 coder** |
| decode-during-compute | **Marlin** (IST-DASLab): FP16×INT4 **fused dequant in the GEMM**, ~4× to batch 16–32, group-128 reshuffled layout | Marlin fuses *fixed-width* dequant; fusing *variable-length entropy* decode into the GEMM is the hard open problem |

**Honest summary (verified):** no single technique is novel. The closest single system is **DFloat11** — and
checking it *confirmed* our distinctiveness rather than undercutting it: DFloat11 decodes **whole tensors**
(not random access), covers **weights only**, and is lossless on **BF16 floats** (not on quantized ints). The
KV works quantize/evict but **don't entropy-code**. The genomics 2026 papers do device-resident random-access
decode but for LZ77 genomics, not LLM data. So the specific combination — **GPU-resident + random-access +
searchable, over quantized values, across *every* LLM stratum** — appears unoccupied, while every mechanism it
uses is established. What is uncommon is the **unification**: one succinct substrate applied across **every** LLM
stratum — weights, KV, MoE experts, LoRA libraries, prompt caches, context self-index, datasets — token- and
tensor-agnostic, with a documented container format. That combination is the v1 contribution, not any one part.

---

## Where the room for improvement is (v1 → v2), and what to borrow

ChromoFold is v1; the measured gaps point straight at the next levers, each with a **verified** prior art to
borrow from (highest-leverage first).

1. **LUT-based GPU Huffman decode, from DFloat11 — DONE (`gpu_block_huffman.py`).** DFloat11 decodes Huffman
   with SRAM lookup tables + a parallel kernel (20.97× faster than nvCOMP). We built `BlockHuffmanArray`: values
   in fixed-count blocks, canonical Huffman, a `2^maxlen` decode LUT, **one GPU thread per block** so the whole
   array reconstructs in parallel — no prefix-sum needed (fixed values/block ⇒ known output positions). Measured
   on 4 M peaky int4 values: **12–21× faster whole-tensor decode than the wavelet** (e.g. block=64: 3.62 b/val,
   972 M/s vs the wavelet's 65 M/s), and at block ≥ 64 it's *also smaller*; random access survives (decode
   within a block). Wired into `QuantizedWeightStore(coder="block")` (6× faster reconstruct on a real tensor,
   serialises + round-trips). The wavelet stays for *search* (FM-index) — two decode modes, choose per need.
2. **rANS/ANS coder, from NeuZip / nvCOMP-gANS — DONE (`gpu_rans.py`).** Built `BlockRANSArray`: 32-bit rANS
   (ryg-style, 12-bit freq table, 8-bit renorm) cut into fixed-count blocks so decode stays one-thread-per-block
   parallel and per-block random access survives — same shape as the Huffman coder. **Honest measured crossover:
   rANS only wins on LOW-entropy + LARGE blocks.** On a very-skewed int4 stream (H0=0.45) at block=1024 it's
   **0.546 vs Huffman's 1.21 b/val (2.2×)** — it breaks Huffman's 1-bit-per-symbol floor. But it carries a
   fixed 32-bit state *per block*, so at small blocks or multi-bit streams (where Huffman is already near-H0) it
   *loses*. So rANS is the coder for the skewed / bulk-decode regime, Huffman for small-block random access —
   both wired as `QuantizedWeightStore(coder="rans"|"block")`. (Corrects a v1 assumption that rANS is a blanket
   upgrade: it isn't — it's entropy-level, but its per-block overhead and Huffman's near-optimality on multi-bit
   data mean it only pays on low-entropy streams.)
3. **KIVI-style KV quantization, then entropy-code it — DONE (`kv_store.py`, `per_axis=True`).** Adopted
   KIVI/KVQuant's **per-channel Keys, per-token Values** (Keys have outlier channels). Measured on gpt2 KV: it
   **lowers attention error ~3.2× at int4 and ~1.8× at int2** (the gap is larger on models with pronounced
   Key-channel outliers — KIVI's regime), which is what lets you drop bits; int2 KIVI = 1.80 b/val / 8.9× vs
   fp16, lossless over its quant, with attended-only (windowed) decode. It costs some b/val (per-axis scales +
   more-uniform values — the accuracy↔size trade). ChromoFold then entropy-codes the result — **the entropy
   layer no KV method has** — via the block coder. (Honest correction to a v1 guess: per-axis improves
   *accuracy*, not compression; the compression comes from the entropy layer on top.) Next: RotateKV/QuIP#
   rotations to tame outliers before coding.
4. **A lossless-BF16 mode, from DFloat11 / NeuZip — DONE (`lossless_float.py`).** `LosslessFloatStore` splits
   each fp16/bf16 value into a low-entropy **exponent** (block-coded, GPU-decodable, **randomly addressable**)
   and a raw **sign+mantissa** (bit-packed). Exact — no quantization. Measured on a real gpt2 tensor: **bf16 →
   11.32 b/val = 1.41× (matching DFloat11's ~11-bit / ~30%)**, fp16 → 14.29 (5-bit exponent, less to squeeze) —
   both lossless (exact bits back) *with* random access, which DFloat11 (whole-tensor decode) lacks. Needed a
   **length-limited Huffman** (JPEG bl_count redistribution) in the block coder so a 256-symbol skewed exponent
   fits the LUT — which also makes the block coder robust for int8 / any large skewed alphabet.
5. **Better quantizers upstream (GPTQ/AWQ/QuIP#/SpQR/XFP).** v1 uses round-to-nearest; calibration makes low-bit
   *usable*, and SpQR/XFP **sparse outlier side-channels** compose naturally with our entropy layer.
6. **Two-level succinct superblocks in VRAM.** v1 delta-compresses superblocks *on disk*; a resident two-level
   rank (int32 anchors + int16 deltas) shrinks them in VRAM while keeping O(1) — the standard succinct-DS trick.
7. **Decode-in-the-matmul, with Marlin as the template.** Marlin fuses *fixed-width* INT4 dequant into the GEMM
   (~4× to batch 16–32). Fusing ChromoFold's *variable-length* entropy decode into the GEMM is the hard open
   problem and the real "bigger model resident during compute" endgame; a LUT-decode-then-Marlin two-stage is
   the pragmatic first step.
8. **On-GPU BWT construction** (genomics has it) so the FM-index *build* is GPU-resident, not just queries.

---

## So, what can we say about ChromoFold?

- It is **not** a new compression algorithm; it is a **GPU-resident, self-indexing framework** that unifies known
  succinct-structure and entropy-coding techniques so that LLM-shaped data stays **navigable while compressed**.
- Its home is the GPU because the trade it makes — **compute for memory** — is favourable exactly where memory
  (not FLOPs) is the bottleneck.
- v1 proves the unification end-to-end and measures it honestly. The improvements above are real and mostly
  borrowable; there is a lot of headroom, and none of it requires the core framing to change.
