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
5. **Sparse outlier side-channel, from SpQR/XFP — DONE (`weight_store.py`, `outliers=`).** A few large-magnitude
   weights blow up the int4 scale, crushing every other weight toward the zero-point (plain per-tensor int4
   "compresses" to 0.68 b/w precisely *because* it's collapsed — MSE garbage). `QuantizedWeightStore(outliers=p)`
   keeps the top-p% |W| in a sparse fp16 side-channel (sorted int32 index — delta+zlib'd in the container — plus
   fp16 value, ~6 B each), computes the int4 scale from the **non-outliers** (tighter), and parks the outlier
   positions at the zero-point (the mode, keeping the stream entropy-codable). Measured on heavy-tailed weights
   (0.3% injected outliers): **int4 + 1% outliers = 4.53 b/w at MSE 2.36e-5 — *more accurate than int8* (4.48 b/w,
   4.37e-5) at the same size**, and **8.4× lower MSE than int4 group-128** (3.82 b/w) — it fixes the *cause* where
   group scaling only softens it. It also **rescues int3** (per-tensor int3 is unusable; +1% outliers → 15×
   better MSE). Exact at the outlier positions, lossless over the quantized rest, GPU-addressable, serialises.
   **Calibration (AWQ) — built and honestly measured (`awq.py`, `bench_awq.py`, `channel_scale=`).** AWQ's
   per-input-channel scaling is a diagonal (so it *preserves random access*, unlike a rotation); `channel_scale`
   quantizes W·diag(s) and undoes it at dequant, lossless. But on **real gpt2 the measured lever is group-wise
   scaling, not AWQ**: per-tensor int4 PPL ≈ 4168 (broken, "commas") → **group-128 int4 PPL 30.86, near the fp32
   26.62 and fully coherent** (int8 = 29.96), while per-tensor+AWQ stays broken (6842 — scaling salient channels
   blows the single tensor scale) and group-128+AWQ (31.56) ≈ group-128 alone. So the accuracy lever that makes
   int4 usable on gpt2 is **grouping — already a ChromoFold knob** (`group_size`) — and AWQ adds nothing here (an
   honest negative; AWQ's win is model- and salient-channel-dependent). ChromoFold entropy-codes whichever form
   you pick and keeps it addressable.
6. **Two-level succinct superblocks in VRAM — DONE (`gpu_rrr.py`, `GPURRR`).** v1 delta-compresses superblocks
   *on disk* (container-only); a resident **two-level rank** shrinks them in VRAM too. `_two_level` splits each
   int32 cumulative sample into an int32 **anchor** every `K=32` superblocks + a **uint16 delta** per superblock
   (the delta ≤ (K−1)·S·T = 29 760 fits uint16), so the resident sample table drops **1.88×** (16 K→8 K bytes on
   a 2 M-bit plane) with one extra add per query and rank still bit-exact. Honest regime dependence: **negligible
   on balanced planes** (samples are a sliver of ~1 b/bit → 2.7% off the plane) but **real on the skewed planes
   the BWT/FM-index produces** — there the samples are the biggest slice, so it trims **up to ~8.8%** off the
   whole plane (p=0.005). Built standalone in `GPURRR`, then **wired under the full RRR wavelet**
   (`gpu_rrr_wavelet.py`, `_two_level_2d`): every level's per-level samples are two-level 2-D arrays, so
   `access`/`rank` and everything on them — the **FM-index** (`GPURRRFMIndex`) and `weight_store(coder="rrr")` —
   inherit it automatically. Measured on a real Markov BWT the FM-index dropped **6.02 → 5.80 b/tok**, now
   *below* the BWT's H₀ (5.96) that the int32-sample version sat above; access/rank/count/predict_next still
   exact, save/load round-trips. The **same split is now also under the Huffman-class wavelet** `RRRWaveletGPUHuff`
   (all three per-level sample tables — rank, offset, *and* the Huffman class-bit position — two-levelled), so the
   `huffman=True` weight-store default *and* the token self-index (`api.compress(tokens)`) inherit it too: **1.88×
   smaller sample table** on both, access exact, weight and token containers round-trip. Two-level is now
   universal across every RRR-backed structure.
7. **Decode-in-the-matmul, with Marlin as the template — DONE as a Warp proof-of-concept (`gpu_fused_matmul.py`).**
   Marlin fuses *fixed-width* INT4 dequant into the GEMM (~4× to batch 16–32); fusing ChromoFold's *variable-
   length* entropy decode is the hard part, and the **fixed-COUNT block layout is what makes it tractable** — the
   bitstream is contiguous, so a thread seeks to any weight row's bit offset (`block_off`) and decodes that row
   *inline* as it multiplies. `FusedDecodeMatmul.matmul(x)` computes `y = x·Wᵀ` reading the block-Huffman int4
   weights straight from the compressed stream, so **the dequantized (M×K) matrix is never materialised in VRAM**
   — measured **10.6× less resident memory during compute** (2048×2048 int4: 1.58 MB compressed vs 16.78 MB
   dense), result rel-error 1.4e-6 (lossless over the quantization). Honest scope: this Warp kernel *re-decodes*
   W per GEMM (one thread per output column) — a real **compute-for-memory** trade at ~4 GFLOP/s, a proof-of-
   concept, **not** a tensor-core Marlin-class fused kernel (that, and a LUT-decode-to-SRAM-tile-then-Marlin
   two-stage, remain the production endgame).
8. **On-GPU BWT construction — DONE (`gpu_suffix.py`).** Genomics aligners build their FM-index on the GPU; the
   one CPU-bound piece left here was the suffix array (`fm_index.suffix_array`, numpy argsort). `gpu_suffix_array`
   does the same prefix-doubling on the device — each round a 64-bit composite key `(rank[i]<<32)|(rank[i+k]+1)`
   built in a kernel, a `wp.utils.radix_sort_pairs`, then `wp.utils.array_scan` over adjacent-key-difference
   flags to re-rank — **bit-identical to the CPU builder** across random / Markov / repetitive / DNA-like inputs,
   and **17–32× faster** (DNA-like V=4, n=400 K: 262 → 8.1 ms). Wired as `GPURRRFMIndex(build="gpu")`, so a token
   stream now goes raw → searchable self-index **without leaving the GPU** (build *and* query resident). Speedup
   grows with n and alphabet; tiny/degenerate inputs favour the CPU.

---

## So, what can we say about ChromoFold?

- It is **not** a new compression algorithm; it is a **GPU-resident, self-indexing framework** that unifies known
  succinct-structure and entropy-coding techniques so that LLM-shaped data stays **navigable while compressed**.
- Its home is the GPU because the trade it makes — **compute for memory** — is favourable exactly where memory
  (not FLOPs) is the bottleneck.
- v1 proves the unification end-to-end and measures it honestly. The improvements above are real and mostly
  borrowable; there is a lot of headroom, and none of it requires the core framing to change.
- **v2 status: all eight borrow-leads are now built and measured** (LUT block decode, block rANS, KIVI per-axis
  KV, lossless bf16, SpQR outlier side-channel, two-level resident superblocks — wired under the RRR wavelet,
  on-GPU suffix-array/BWT build, and fused decode-in-the-matmul). Each confirmed the same shape: the mechanism is
  borrowed and known, and ChromoFold's distinctive move (GPU-resident + random-access + searchable, across every
  stratum, over quantized/entropy-coded values) survives each one. The honestly-remaining work is *engineering
  depth*, not new levers: calibration quantizers (GPTQ/AWQ) upstream of the outlier channel, the two-level split
  under the Huffman-class wavelet too, and a tensor-core Marlin-class version of the fused decode-GEMM.
