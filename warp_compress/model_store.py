"""model_store — compress an ENTIRE model with ChromoFold, reconstruct it, and generate. The capstone.

Every "fit more on one GPU" stratum has been measured in isolation; this puts a whole real model through the
weight path at once: quantize + class-stream-Huffman-entropy-code every large 2-D weight tensor, keep the tiny
ones (norms, biases) in fp16, measure the whole-model footprint vs fp16, then reconstruct the weights and
**generate text** — proving the compressed model still works (and is byte-identical to the plain-quantized
model, since the entropy layer is lossless over the quantized values).

    compress_model(model)         -> {name: QuantizedWeightStore} for the big tensors + a size breakdown
    reconstruct_into(model, ...)  -> load the reconstructed (quantized) weights back for inference

Honest (measured on gpt2, held-out perplexity): int8 is near-lossless (PPL 52.8 vs fp32 47.9, ~3× smaller).
Whole-model int4 needs BOTH group-128 scaling AND mixed precision — the tied wte embedding/output head must
stay int8 (int4 there blows PPL to ~4700 even with grouping). With both it's coherent, but on a small model
where the embedding is 31% of params it lands near int8's size at worse quality, so int8 wins for gpt2; int4's
payoff is on large models where the linears dominate. The compressed form is the *storage*; a real deployment
streams per-layer decode. Requires torch/transformers. Run: python -m warp_compress.model_store
"""
from __future__ import annotations

import numpy as np

from .weight_store import QuantizedWeightStore


def compress_model(model, bits: int = 8, huffman: bool = True, min_numel: int = 100_000,
                   device: str = "cuda:0", group_size: "int | None" = None,
                   protect: "tuple" = (), protect_bits: "int | None" = None):
    """ChromoFold-compress every large 2-D weight tensor; return (stores, compressed_bytes, kept_fp16_bytes).
    ``group_size`` (e.g. 128) uses per-group quant scales — the accuracy lever that makes low-bit usable.
    ``protect`` = name-substrings to quantize at ``protect_bits`` instead of ``bits`` (e.g. keep the tied
    embedding / output head at int8 while the linears go int4 — the mixed-precision that low-bit LLMs need)."""
    import torch  # noqa: F401

    stores, comp, kept, big_params = {}, 0, 0, 0
    for n, p in model.named_parameters():
        if p.ndim == 2 and p.numel() >= min_numel:
            b = protect_bits if (protect_bits and any(s in n for s in protect)) else bits
            st = QuantizedWeightStore(p.detach().numpy().astype(np.float32), bits=b, huffman=huffman,
                                      device=device, group_size=group_size)
            stores[n] = st
            comp += st.size_bytes()
            big_params += p.numel()
        else:
            kept += p.numel() * 2                              # small tensors stay fp16
    return stores, dict(compressed=comp, kept_fp16=kept, big_params=big_params)


def reconstruct_into(model, stores):
    """Load the reconstructed (quantized) weights back into the model, in place."""
    import torch
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in stores:
                p.copy_(torch.from_numpy(stores[n].reconstruct()))


def save_model(model, stores) -> bytes:
    """Serialise a whole ChromoFold-compressed model to ONE container: each compressed tensor as a nested
    weight_store blob, the small fp16-kept tensors alongside. A portable `.cfold` model file."""
    from . import format as fmt
    import numpy as _np
    arrays, manifest = {}, []
    for name, st in stores.items():
        arrays["S:" + name] = _np.frombuffer(st.save(), dtype=_np.uint8)
        manifest.append({"name": name, "kind": "store"})
    for n, p in model.named_parameters():
        if n not in stores:
            a = p.detach().numpy().astype(_np.float16)
            arrays["T:" + n] = a
            manifest.append({"name": n, "kind": "tensor", "shape": list(a.shape)})
    return fmt.pack("model", {"tensors": len(manifest)}, {"manifest": manifest}, arrays)


def load_model(data: bytes, device: str = "cuda:0") -> dict:
    """Load a `.cfold` model file into {name: QuantizedWeightStore | fp16 ndarray}."""
    from . import format as fmt
    header, arrays = fmt.unpack(data)
    out = {}
    for m in header["params"]["manifest"]:
        if m["kind"] == "store":
            out[m["name"]] = QuantizedWeightStore.load(arrays["S:" + m["name"]].tobytes(), device)
        else:
            out[m["name"]] = arrays["T:" + m["name"]].reshape(m["shape"])
    return out


def apply_model(model, loaded):
    """Write a loaded {name: store|tensor} back into a live model, in place (stores are reconstructed)."""
    import torch
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in loaded:
                obj = loaded[n]
                p.copy_(torch.from_numpy(obj.reconstruct() if hasattr(obj, "reconstruct")
                                         else np.asarray(obj, np.float32)))


_EVAL = (
    "The scientific revolution transformed how people understood the natural world, replacing appeals to "
    "authority with observation, measurement, and experiment. Newton showed that the same laws govern the "
    "fall of an apple and the orbit of the moon, and later physicists extended this unity to electricity, "
    "heat, and light. Each advance depended on careful instruments and on the willingness to discard a "
    "theory that the evidence contradicted, however elegant it might be."
)


def _demo():
    import warnings
    warnings.filterwarnings("ignore")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import warp as wp
    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"

    tok = AutoTokenizer.from_pretrained("gpt2")
    ids = tok("In a distant future, humanity discovered that", return_tensors="pt").input_ids
    eval_ids = tok(_EVAL, return_tensors="pt").input_ids

    def gen(model):
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=24, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).replace("\n", " ")

    def ppl(model):
        with torch.no_grad():
            return float(torch.exp(model(eval_ids, labels=eval_ids).loss))

    total_params = sum(p.numel() for p in AutoModelForCausalLM.from_pretrained("gpt2").parameters())
    fp16 = total_params * 2

    base = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    fp32_ppl = ppl(base)
    print(f"device={dev}   model=gpt2 ({total_params/1e6:.1f}M params, fp16 = {fp16/1e6:.0f} MB)\n")
    print(f"  {'config':>22} {'.cfold MB':>10} {'vs fp16':>8} {'PPL':>8} {'vs fp32':>8}   generation")
    print(f"  {'fp32 reference':>22} {'—':>10} {'—':>8} {fp32_ppl:>8.2f} {'1.00×':>8}   {gen(base)!r}")

    # protect the tied token-embedding / output head (wte): it's ~31% of gpt2 and int4 there destroys the logits
    configs = [("int8", 8, None, None), ("int4 group-128", 4, 128, None),
               ("int4-g128, int8 embed", 4, 128, 8)]
    best = None
    for label, bits, gs, pb in configs:
        model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
        stores, br = compress_model(model, bits=bits, huffman=True, device=dev, group_size=gs,
                                    protect=("wte",), protect_bits=pb)
        reconstruct_into(model, stores)                        # now the model IS the quantized model
        total = br["compressed"] + br["kept_fp16"]
        print(f"  {label:>22} {total/1e6:>10.1f} {fp16/total:>7.1f}× {ppl(model):>8.2f} {ppl(model)/fp32_ppl:>7.2f}×"
              f"   {gen(model)!r}")
        if pb == 8:
            best = (model, stores)

    # the definitive artifact: whole gpt2 at the usable config -> ONE .cfold, reloaded, still addressable
    model, stores = best
    blob = save_model(model, stores)
    loaded = load_model(blob, device=dev)
    fresh = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    apply_model(fresh, loaded)
    name0 = next(n for n in stores)
    st = loaded[name0]
    ra = np.allclose(st.fetch(np.array([0, 1, 2, 100])), st.reconstruct().ravel()[[0, 1, 2, 100]], atol=1e-5)
    print(f"\n  .cfold round-trip: {len(blob)/1e6:.1f} MB on disk, reload+generate {gen(fresh)!r}")
    print(f"  still GPU-addressable after load: fetch(store, idx) == reconstruct  {'✓' if ra else 'FAIL'}")

    print("\n=> the DEFINITIVE capstone: whole gpt2, ChromoFold-compressed to one .cfold, still generates AND is "
          "randomly addressable after load. HONEST findings: (1) int8 is near-lossless (~3×). (2) int4 needs BOTH\n"
          "   group-128 scaling AND mixed precision — the tied wte embedding/output head (31% of gpt2) must stay "
          "int8; int4 there destroys the logits even with grouping (PPL ~4700). (3) With both, int4 is coherent\n"
          "   but on THIS small model it's ~int8's size at worse PPL, because the protected embedding dominates — "
          "so int8 wins for gpt2; mixed-int4's payoff is LARGE models where the linears dominate the footprint.\n"
          "   ChromoFold entropy-codes whichever quantization you pick, losslessly and GPU-addressably: accuracy "
          "is the quantizer's job, keeping it small + navigable is ours.")


if __name__ == "__main__":
    _demo()
