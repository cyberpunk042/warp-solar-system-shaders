# CLAUDE.md — ChromoFold Python/Warp prototype

> Onboarding for an AI coding session in the **prototype** repo. The native C++/CUDA engine is a sibling repo,
> `~/chromoFold` — see its [`CLAUDE.md`](../chromoFold/CLAUDE.md) and
> [`docs/PROJECT_SYNC.md`](../chromoFold/docs/PROJECT_SYNC.md).

## What this is

`warp_compress/` is the **research surface, correctness oracle, and performance floor** for **ChromoFold** — a
GPU-resident, random-access, searchable succinct-data-structure runtime for LLM data. It validates ideas cheaply
in Python + NVIDIA Warp and holds the honest, end-to-end measurements. The hot primitives are then ported to the
native engine (`~/chromoFold`), which must reproduce this repo's output **bit-for-bit**.

(The `warp_shaders/`, `thread.py`, and genome-visualization code are the original DNA-folding *visualization* —
branding, not the compressor. `warp_compress/` is the actual system.)

## Run

```sh
.venv/bin/python -m pytest tests/ -q                 # the suite (341 tests; needs the venv)
.venv/bin/python -m warp_compress.<module>           # a module demo (many print measured results)
```

The `.venv` has warp 1.15, torch 2.13 (CPU), transformers 5.14. Cached models: gpt2, Qwen2.5-0.5B/1.5B-Instruct.
Llama-3.2 is **gated** (no token on this box) — use Qwen2.5 as the ungated modern-GQA substitute. For offline
runs export `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. Real-model benchmarks are slow on CPU — run in background.

## What's here (high level)

- **Succinct GPU primitives:** `gpu_wavelet` (packed access/rank), `gpu_rrr` (RRR bitvector), `gpu_rrr_wavelet`
  (RRR wavelet + FM-index), `gpu_rrr_huffman`, `gpu_block_huffman`, `gpu_rans`, `gpu_suffix` (GPU suffix array),
  `gpu_delta`, `gpu_fused_matmul`.
- **LLM strata:** `weight_store` (quantized + entropy + outliers + AWQ channel-scale), `kv_store` (KIVI per-axis
  + windowed), `moe_store`, `lora_library`, `prompt_cache`, `dedup`, `multi_seed`, `model_store` (whole-model),
  `hf_cache` (drop-in transformers KV cache), `spec_decode`, `awq`, `autotune`, `format` (`.cfold` container).
- **Benchmarks:** `bench_*.py` (weights, KV scaling, frontier, stack, dataset dedup, AWQ).
- **Docs:** `docs/chromofold*.md`, `docs/_PDFs/ChromoFold_Research_Brief.pdf`.

## Package / distribute

This repo ships an installable **`chromofold`** package (`pyproject.toml` + the `chromofold/` public-API shim
over `warp_compress`) for Hugging Face + sovereign / on-prem use:

```sh
pip install -e .            # or  pip install chromofold  /  chromofold[torch]
python -c "import chromofold as cf; print(cf.__version__)"   # offline, torch-free
```

`import chromofold` does **no network I/O** and does not import torch (the transformers KV cache loads lazily).
The public surface is `compress` / `Artifact` / `QuantizedWeightStore` / `KVCacheStore` / `MoEExpertStore` /
`ChromoFoldCache`. See [`INTEGRATION.md`](INTEGRATION.md) for the HF and air-gapped-deployment guides. The
visualization code is excluded from the wheel; `warp_compress` remains the internal reference implementation.

## Discipline (shared with the native engine)

- **Measured, not asserted.** Every claim carries its numbers + an honest baseline. **Report negatives** (e.g.
  AWQ did not help gpt2; grouping did) — the intellectual honesty is the brand.
- **Lossless over the chosen quantization.** Quantization is the lossy lever; the entropy + index layer is
  bit-exact on top and randomly addressable.
- When a ported primitive changes here, follow the sync checklist in
  [`../chromoFold/docs/PROJECT_SYNC.md`](../chromoFold/docs/PROJECT_SYNC.md): bump the reference format version if
  the on-wire layout changed, regenerate the golden vectors, and keep the native benches bit-identical.

## Persistent memory

Session memory lives in `~/.claude/projects/-home-jfortin-warp-solar-system-shaders/memory/` (indexed by
`MEMORY.md`): `genome-compression-research.md` (the full build log), `chromofold-native-engine.md` (go-to-market
+ the native engine at `~/chromoFold`).

## Commit convention

Branch off `main` only if asked; otherwise commit to `main`. End messages with:
`Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
