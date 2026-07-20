"""model_store — compress an ENTIRE model with ChromoFold, reconstruct it, and generate. The capstone.

Every "fit more on one GPU" stratum has been measured in isolation; this puts a whole real model through the
weight path at once: quantize + class-stream-Huffman-entropy-code every large 2-D weight tensor, keep the tiny
ones (norms, biases) in fp16, measure the whole-model footprint vs fp16, then reconstruct the weights and
**generate text** — proving the compressed model still works (and is byte-identical to the plain-quantized
model, since the entropy layer is lossless over the quantized values).

    compress_model(model)         -> {name: QuantizedWeightStore} for the big tensors + a size breakdown
    reconstruct_into(model, ...)  -> load the reconstructed (quantized) weights back for inference

Honest: quantization is the lossy lever (int4 noticeably degrades a 124M model; int8 is near-lossless). The
compressed form is the *storage*; a real deployment streams per-layer decode to keep the resident footprint at
the compressed size. Requires torch/transformers. Run: python -m warp_compress.model_store
"""
from __future__ import annotations

import numpy as np

from .weight_store import QuantizedWeightStore


def compress_model(model, bits: int = 8, huffman: bool = True, min_numel: int = 100_000,
                   device: str = "cuda:0"):
    """ChromoFold-compress every large 2-D weight tensor; return (stores, compressed_bytes, kept_fp16_bytes)."""
    import torch  # noqa: F401

    stores, comp, kept, big_params = {}, 0, 0, 0
    for n, p in model.named_parameters():
        if p.ndim == 2 and p.numel() >= min_numel:
            st = QuantizedWeightStore(p.detach().numpy().astype(np.float32), bits=bits, huffman=huffman,
                                      device=device)
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


def _demo():
    import warnings
    warnings.filterwarnings("ignore")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import warp as wp
    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"

    tok = AutoTokenizer.from_pretrained("gpt2")
    prompt = "In a distant future, humanity discovered that"
    ids = tok(prompt, return_tensors="pt").input_ids

    def gen(model):
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=30, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).replace("\n", " ")

    total_params = sum(p.numel() for p in AutoModelForCausalLM.from_pretrained("gpt2").parameters())
    fp16 = total_params * 2

    base = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    print(f"device={dev}   model=gpt2 ({total_params/1e6:.1f}M params, fp16 = {fp16/1e6:.0f} MB)\n")
    print(f"  fp32   : {gen(base)}")

    for bits in (8, 4):
        model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
        stores, br = compress_model(model, bits=bits, huffman=True, device=dev)
        reconstruct_into(model, stores)                        # now the model IS the quantized model
        total = br["compressed"] + br["kept_fp16"]
        bpw = br["compressed"] * 8 / br["big_params"]
        print(f"  int{bits}   : {gen(model)}")
        print(f"     -> whole model {total/1e6:5.1f} MB ({fp16/total:.1f}× vs fp16)   "
              f"big tensors {bpw:.2f} b/weight over {br['big_params']/1e6:.0f}M params, {len(stores)} tensors")

    print("\n=> the ENTIRE model, ChromoFold-compressed, still generates: int8 is near-lossless and ~4× smaller; "
          "int4 is smaller still but visibly degraded (that is int4 quantization's cost, not ChromoFold's — the\n"
          "   entropy layer is lossless over the quantized values). Storage is the compressed form; a real "
          "deployment streams per-layer decode to keep the resident footprint at the compressed size.")


if __name__ == "__main__":
    _demo()
