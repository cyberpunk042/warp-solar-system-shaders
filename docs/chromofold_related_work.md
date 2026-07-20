# What ChromoFold is, and who came before — the honest lineage

*This is the intellectually-honest companion to the positioning doc. ChromoFold (v1) reuses a lot of known
ideas; being clear about that is a feature — it means the direction is validated, and it points at where the
real room for improvement is. Citations here are named to the best of memory; verify exact references before
publishing.*

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

## Prior art — every ingredient has a lineage

| ChromoFold piece | Prior art it descends from | How ours differs / overlaps |
|---|---|---|
| GPU-resident (de)compression | **nvCOMP** (LZ4, Snappy, GDeflate, **rANS/ANS**, bitcomp, cascaded) | nvCOMP is streaming/block; ours adds per-element **random access + search**, token/tensor-aware |
| entropy-coding quantized weights | **Deep Compression** (Han et al. 2016: prune→quantize→Huffman); **DFloat11**, **NeuZip** (lossless entropy code + GPU decode) | same lineage; ours is one substrate (RRR/Huffman wavelet) shared with the index/KV/MoE paths |
| the lossy quantizer we compose with | **GPTQ**, **AWQ**, **bitsandbytes / QLoRA NF4**, **SmoothQuant**, **SpQR** | we don't quantize better; we entropy-code + address the quantized stream losslessly |
| KV-cache shrink | **KVQuant**, **KIVI**, **H2O**; **PagedAttention/vLLM** (memory mgmt, not compression) | ours: quantize + entropy-code + **attended-only decode** in one format |
| the self-index (rank/select) | **FM-index** (Ferragina–Manzini), **wavelet trees** (Grossi–Gupta–Vitter; Claude–Navarro), **RRR** bitvectors | textbook succinct structures, put on the GPU over token streams |
| GPU FM-index / BWT | genomics read aligners: **nvBIO**, **SOAP3-dp**, **CUSHAW**, **BarraCUDA** | same machinery, retargeted from DNA reads to LLM tokens |
| n-gram LM from a suffix index | **infini-gram** (Liu et al. 2024); classic suffix-array LMs | we rebuilt this independently (`predict_next`) before naming it |
| reference/delta clusters | delta/dictionary compression, VCF-style variant encoding, grammar-compressed indexes | ours keeps GPU O(depth) random access to any member |
| entropy coder | **Huffman**; **rANS/ANS** (Duda 2013, used in zstd-FSE, nvCOMP) | v1 uses RRR + canonical Huffman; rANS is an obvious v2 |

**Honest summary:** no single technique here is novel. FM-index = genomics; Huffman-coded quantized weights =
Deep Compression / DFloat11; the suffix-index LM = infini-gram; GPU codecs = nvCOMP. What is uncommon is the
**unification**: one GPU-resident, random-access, *searchable* succinct substrate applied across **every** LLM
stratum — weights, KV, MoE experts, LoRA libraries, prompt caches, context self-index, datasets — token- and
tensor-agnostic, with a documented container format. That combination is the v1 contribution, not any one part.

---

## Where the room for improvement is (v1 → v2), and what to borrow

ChromoFold is v1; the measured gaps point straight at the next levers, and most have a prior art to borrow from.

1. **rANS/ANS instead of (or beside) RRR+Huffman.** ANS is near-optimal and streams fast; nvCOMP already has a
   GPU ANS. It would tighten the entropy stage — at some cost to O(1) random access (ANS is sequential), so it
   fits the *archive* end, RRR the *random-access* end. A hybrid per-section coder is the honest design.
2. **Two-level succinct structures.** v1 stores int32 superblocks (compressed on disk, §format). A proper
   two-level rank (int32 anchors + int16 in-block deltas) shrinks them **in VRAM too** while keeping O(1) — the
   standard succinct-DS trick we haven't yet applied resident.
3. **Better quantizers upstream.** GPTQ/AWQ/SpQR calibration makes low-bit *usable* (v1 uses round-to-nearest);
   ChromoFold entropy-codes whatever they emit. SpQR-style outlier side-channels compose naturally.
4. **Learned / context-mixing entropy models** for the id stream (the "compression is prediction" view), where
   the FM-index already gives a predictor — a bridge between the index and the coder.
5. **Product / vector quantization** for weights and KV (share a codebook across a tensor), which changes what
   the entropy layer sees.
6. **On-GPU BWT construction** (genomics has it) so the FM-index build itself is GPU-resident, not just queries.
7. **A live decode-in-the-matmul path** — fuse dequant+entropy-decode into the GEMM so the resident footprint
   stays compressed *during* compute, not just at rest (the real "fit a bigger model" endgame).

---

## So, what can we say about ChromoFold?

- It is **not** a new compression algorithm; it is a **GPU-resident, self-indexing framework** that unifies known
  succinct-structure and entropy-coding techniques so that LLM-shaped data stays **navigable while compressed**.
- Its home is the GPU because the trade it makes — **compute for memory** — is favourable exactly where memory
  (not FLOPs) is the bottleneck.
- v1 proves the unification end-to-end and measures it honestly. The improvements above are real and mostly
  borrowable; there is a lot of headroom, and none of it requires the core framing to change.
