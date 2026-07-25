"""bench_levers — put every weight-compression lever through a real model and report perplexity vs bits/weight.

The individual lever modules (`weight_store`, `vq_store`, `rvq_store`, `lowrank_store`, `pq_subspace`) each carry
their own *synthetic* rate-distortion measurement. This harness is the **model-gated** counterpart the research
doc calls for: it patches each Linear weight of a real transformer with a lever's reconstruction and measures the
end-to-end quality signal that actually matters — **perplexity** — against the lever's **average bits/weight**.
The point is to answer "which lever, at which budget, costs the least real quality?" on gpt2 / Qwen rather than on
Gaussian toys.

Two halves, split by dependency so the correctness core runs anywhere:

  * `apply_lever(W, name, **kw) -> (recon, bits_per_weight)` — **torch-free**. Reuses the exact stores shipped in
    this repo, returns the lossy reconstruction + the measured rate. Verified here (numpy + warp only), no model.
  * `main()` — **torch-gated**. Loads a cached HF model, swaps each Linear (and gpt2 `Conv1D`) weight for its
    lever reconstruction, and prints a perplexity-vs-rate table per lever. Skips cleanly with a clear message when
    torch / transformers / a cached model are absent (this container has none — see CLAUDE.md: model hosts are
    proxy-blocked here), so it produces **no numbers in this sandbox** but is correct and ready on a GPU box.

Honesty contract (repo doctrine): the harness reports whatever perplexity comes out, including regressions — a
lever that saves bits but wrecks perplexity is a reported negative, not a hidden one. The bits/weight is the
*measured* store size (entropy-coded index + fp16 codebooks), not the nominal width.

    python -m warp_compress.bench_levers                 # runs the model eval, or explains what's missing
    python -m warp_compress.bench_levers --selftest      # torch-free: verifies apply_lever on every lever
    python -m warp_compress.bench_levers --model gpt2 --levers int4,pq,subspace_pq --tokens 4096
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

# ------------------------------------------------------------------------------------------------------------------
# torch-free core: each lever as (reconstruction, measured bits/weight). Reuses the shipped stores verbatim.
# ------------------------------------------------------------------------------------------------------------------

# A lever may not fit an arbitrary weight shape (PQ needs size % subdim == 0; subspace/low-rank need 2-D with
# in % subdim == 0). `apply_lever` raises LeverIncompatible for those so the model loop can SKIP that layer for
# that lever instead of crashing — and the reported rate averages only over layers a lever could actually take.


class LeverIncompatible(ValueError):
    """The lever cannot encode a weight of this shape (e.g. divisibility constraint) — skip it for this layer."""


def _int(W, bits, outliers=0.0, **kw):
    from .weight_store import QuantizedWeightStore
    st = QuantizedWeightStore(np.asarray(W, np.float32), bits=bits, huffman=True, outliers=outliers, device=kw.get("device", "cpu"))
    return st.reconstruct().reshape(W.shape), st.bits_per_weight()


def _pq(W, subdim=4, codebook_bits=8, **kw):
    from .vq_store import ProductQuantizedWeightStore
    if W.size % subdim != 0:
        raise LeverIncompatible(f"size {W.size} not a multiple of subdim {subdim}")
    st = ProductQuantizedWeightStore(np.asarray(W, np.float32), subdim=subdim, codebook_bits=codebook_bits,
                                     device=kw.get("device", "cpu"), seed=kw.get("seed", 0),
                                     kmeans_iters=kw.get("kmeans_iters", 10))
    return st.reconstruct().reshape(W.shape), st.bits_per_weight()


def _rvq(W, subdim=4, codebook_bits=4, stages=2, **kw):
    from .rvq_store import ResidualVQWeightStore
    if W.size % subdim != 0:
        raise LeverIncompatible(f"size {W.size} not a multiple of subdim {subdim}")
    st = ResidualVQWeightStore(np.asarray(W, np.float32), subdim=subdim, codebook_bits=codebook_bits, stages=stages,
                               device=kw.get("device", "cpu"), seed=kw.get("seed", 0),
                               kmeans_iters=kw.get("kmeans_iters", 10))
    return st.reconstruct().reshape(W.shape), st.bits_per_weight()


def _lowrank(W, rank=16, residual="int", **kw):
    from .lowrank_store import LowRankWeightStore     # SVD-based: deterministic, no seed/kmeans_iters
    if W.ndim != 2:
        raise LeverIncompatible("low-rank needs a 2-D weight")
    st = LowRankWeightStore(np.asarray(W, np.float32), rank=rank, residual=residual, device=kw.get("device", "cpu"))
    return st.reconstruct().reshape(W.shape), st.bits_per_weight()


def _hadamard(W, bits=4, **kw):
    from .hadamard_store import HadamardQuantStore    # incoherence: seeded Hadamard rotation, then int-quantize
    if W.ndim != 2:
        raise LeverIncompatible("hadamard-quant needs a 2-D weight")
    st = HadamardQuantStore(np.asarray(W, np.float32), bits=bits, device=kw.get("device", "cpu"),
                            seed=kw.get("seed", 0))
    return st.reconstruct().reshape(W.shape), st.bits_per_weight()


def _subspace_pq(W, subdim=4, codebook_bits=8, **kw):
    from .pq_subspace import SubspacePQWeightStore
    if W.ndim != 2:
        raise LeverIncompatible("subspace-PQ needs a 2-D weight")
    if W.shape[1] % subdim != 0:
        raise LeverIncompatible(f"in {W.shape[1]} not a multiple of subdim {subdim}")
    st = SubspacePQWeightStore(np.asarray(W, np.float32), subdim=subdim, codebook_bits=codebook_bits,
                               device=kw.get("device", "cpu"), seed=kw.get("seed", 0),
                               kmeans_iters=kw.get("kmeans_iters", 10))
    return st.reconstruct().reshape(W.shape), st.bits_per_weight()


# name -> (fn, default kwargs). Rates: int8~8, int4~4, int4_outliers~4.5, pq(subdim4,8b)~2, rvq(2x4b subdim4)~2,
# lowrank rank/residual dependent, subspace_pq(subdim4,8b)~2.
LEVERS = {
    "int8":          (_int,         dict(bits=8)),
    "int4":          (_int,         dict(bits=4)),
    "int4_outliers": (_int,         dict(bits=4, outliers=0.01)),
    "pq":            (_pq,          dict(subdim=4, codebook_bits=8)),
    "rvq":           (_rvq,         dict(subdim=4, codebook_bits=4, stages=2)),
    "lowrank":       (_lowrank,     dict(rank=32, residual="int")),
    "subspace_pq":   (_subspace_pq, dict(subdim=4, codebook_bits=8)),
    "hadamard":      (_hadamard,    dict(bits=4)),
}


def apply_lever(W, name, **overrides):
    """Encode weight `W` with lever `name`, return (reconstruction float32 same shape, measured bits/weight).

    Raises LeverIncompatible when the lever can't take this shape. `overrides` merge over the lever defaults
    (e.g. subdim, codebook_bits, rank, device, seed, kmeans_iters)."""
    if name not in LEVERS:
        raise KeyError(f"unknown lever {name!r}; known: {sorted(LEVERS)}")
    fn, defaults = LEVERS[name]
    kw = {**defaults, **overrides}
    W = np.asarray(W, np.float32)
    recon, bpw = fn(W, **kw)
    return np.asarray(recon, np.float32), float(bpw)


# ------------------------------------------------------------------------------------------------------------------
# torch-free self-test — runs HERE (numpy + warp), no model. Proves apply_lever on every lever + the skip path.
# ------------------------------------------------------------------------------------------------------------------

def _selftest() -> int:
    rng = np.random.default_rng(0)
    W = ((rng.standard_normal((256, 128)) @ rng.standard_normal((128, 128))) / 128
         + rng.standard_normal((256, 128)) / np.sqrt(128)).astype(np.float32)   # low-rank + noise (model-like)
    denom = float((W ** 2).mean())
    print(f"apply_lever self-test  (W {W.shape}, torch-free)\n")
    print(f"  {'lever':<15}{'bits/wt':>9}{'recon NMSE':>13}   shape-ok")
    ok = True
    for name in LEVERS:
        recon, bpw = apply_lever(W, name)
        nmse = float(((recon - W) ** 2).mean() / denom)
        good = recon.shape == W.shape and np.isfinite(recon).all() and 0.0 < bpw < 40.0 and nmse < 1.0
        ok = ok and good
        print(f"  {name:<15}{bpw:9.2f}{nmse:13.2e}   {'OK' if good else 'FAIL'}")

    # the skip path: a shape no divisibility-constrained lever can take (odd size, prime-ish in-dim)
    Wbad = rng.standard_normal((7, 5)).astype(np.float32)      # size 35 (odd), in=5
    skipped = []
    for name in ("pq", "rvq", "subspace_pq"):
        try:
            apply_lever(Wbad, name)
        except LeverIncompatible:
            skipped.append(name)
    good = skipped == ["pq", "rvq", "subspace_pq"]
    ok = ok and good
    print(f"\n  incompatible shape {Wbad.shape}: correctly skipped {skipped}  {'OK' if good else 'FAIL'}")
    print("\n=> apply_lever reconstructs every lever + raises LeverIncompatible on unfittable shapes." if ok
          else "\n=> SELF-TEST FAILED")
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------------------------------
# model-gated eval — perplexity vs bits/weight per lever. torch-gated; skips cleanly when unavailable.
# ------------------------------------------------------------------------------------------------------------------

@dataclass
class LeverResult:
    lever: str
    ppl: float
    bits_per_weight: float
    layers_encoded: int
    layers_skipped: int
    weights_encoded: int


def _iter_linear_weights(model):
    """Yield (module, attr, W_np, transpose_back) for every 2-D weight matrix in a HF causal-LM.

    gpt2 uses `transformers.pytorch_utils.Conv1D` (weight is (in, out)); everything else is `nn.Linear`
    (weight is (out, in)). We compress in (out, in) orientation and write back in the module's native layout so
    the forward pass is unchanged apart from the lossy values."""
    import torch.nn as nn
    try:
        from transformers.pytorch_utils import Conv1D
    except Exception:                                           # older/newer transformers layout
        Conv1D = ()
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            yield mod, mod.weight, False                        # (out, in) already
        elif Conv1D and isinstance(mod, Conv1D):
            yield mod, mod.weight, True                         # (in, out) -> transpose to (out, in) to encode


def _perplexity(model, input_ids) -> float:
    import torch
    with torch.no_grad():
        loss = model(input_ids, labels=input_ids).loss
    return float(torch.exp(loss))


def _eval_model(model_name, lever_names, n_tokens, seed, device, kmeans_iters):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device).eval()

    # a fixed evaluation text (repeated to fill the token budget) — deterministic, no dataset download needed
    seed_text = (
        "The study of compression is the study of what can be predicted. A model that compresses well is a model "
        "that understands. In this repository we measure, we do not assert; every claim carries its numbers and an "
        "honest baseline, and negatives are reported as loudly as wins. "
    )
    ids = tok(seed_text * 64, return_tensors="pt").input_ids[:, :n_tokens].to(device)

    base_ppl = _perplexity(model, ids)
    print(f"model={model_name}  device={device}  eval tokens={ids.shape[1]}")
    print(f"baseline (fp32) perplexity = {base_ppl:.3f}\n")
    print(f"  {'lever':<15}{'ppl':>9}{'Δppl%':>9}{'bits/wt':>9}{'layers':>9}{'skipped':>9}")

    targets = list(_iter_linear_weights(model))
    originals = [(mod, w.detach().clone()) for mod, w, _ in targets]

    results = []
    for name in lever_names:
        enc = skipped = 0
        tot_bits = tot_wts = 0
        with torch.no_grad():
            for (mod, w, transpose), (_, w0) in zip(targets, originals):
                W = w0.T.cpu().numpy() if transpose else w0.cpu().numpy()   # encode in (out, in)
                try:
                    recon, bpw = apply_lever(W, name, seed=seed, device="cpu", kmeans_iters=kmeans_iters)
                except LeverIncompatible:
                    w.copy_(w0)                                 # leave this layer at full precision
                    skipped += 1
                    continue
                recon_t = torch.from_numpy(recon.T if transpose else recon).to(w.dtype).to(device)
                w.copy_(recon_t)
                enc += 1
                tot_bits += bpw * W.size
                tot_wts += W.size
        ppl = _perplexity(model, ids)
        for (mod, w, _), (_, w0) in zip(targets, originals):
            w.copy_(w0)                                          # restore fp32 before the next lever
        avg_bpw = tot_bits / max(1, tot_wts)
        dppl = 100.0 * (ppl - base_ppl) / base_ppl
        print(f"  {name:<15}{ppl:9.3f}{dppl:9.2f}{avg_bpw:9.2f}{enc:9d}{skipped:9d}")
        results.append(LeverResult(name, ppl, avg_bpw, enc, skipped, tot_wts))

    print("\n=> lower bits/wt with the smallest Δppl% wins. Encoded-only layers count toward the rate; skipped\n"
          "   (shape-incompatible) layers stay fp32, so a lever that skips many layers shows an optimistic rate —\n"
          "   read bits/wt together with the 'layers' column. Report negatives: a big +Δppl% is a real cost.")
    return base_ppl, results


def main(argv=None):
    p = argparse.ArgumentParser(description="perplexity vs bits/weight for every weight-compression lever")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--levers", default=",".join(LEVERS), help="comma list; default all")
    p.add_argument("--tokens", type=int, default=2048, help="eval token budget")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="torch device (default: cuda if available else cpu)")
    p.add_argument("--kmeans-iters", type=int, default=10)
    p.add_argument("--selftest", action="store_true", help="torch-free: verify apply_lever on every lever, then exit")
    args = p.parse_args(argv)

    if args.selftest:
        raise SystemExit(_selftest())

    try:
        import torch
        from transformers import AutoModelForCausalLM  # noqa: F401
    except Exception as e:
        print("bench_levers model eval needs torch + transformers + a cached model — none available here.\n"
              f"  ({type(e).__name__}: {e})\n"
              "  This sandbox has no torch and the model hosts are proxy-blocked (see CLAUDE.md), so the eval\n"
              "  produces no numbers here. The torch-free core is verified with:\n"
              "      python -m warp_compress.bench_levers --selftest\n"
              "  On a GPU box with a cached model (gpt2 / Qwen2.5), this same command prints the ppl-vs-rate table.")
        return

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    lever_names = [s.strip() for s in args.levers.split(",") if s.strip()]
    unknown = [n for n in lever_names if n not in LEVERS]
    if unknown:
        raise SystemExit(f"unknown levers {unknown}; known: {sorted(LEVERS)}")
    try:
        _eval_model(args.model, lever_names, args.tokens, args.seed, device, args.kmeans_iters)
    except Exception as e:
        if "offline" in str(e).lower() or "connect" in str(e).lower() or "not a local folder" in str(e).lower():
            print(f"could not load model {args.model!r} (not cached, and downloads are offline/blocked here):\n"
                  f"  {type(e).__name__}: {e}\n"
                  "  Pre-cache the model on a networked box, then run this offline "
                  "(HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1).")
        else:
            raise


if __name__ == "__main__":
    main()
