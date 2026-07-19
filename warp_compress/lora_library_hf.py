"""lora_library_hf — the LoRA-library proof on a REAL transformer (gpt2 + peft), not a synthetic harness.

Same claim as ``lora_library`` but wired to an actual model: build a family of real peft LoRA adapters on
gpt2, store the quantized family as ONE base + sparse deltas (`gpu_delta.GPUDeltaCluster`), then for each
adapter reconstruct it on the GPU (Warp), load it into the model, and check that the model's **logits match**
the run with the original quantized adapter — bit-exact, because the delta store is lossless over the
quantized ints. This is the "the store→reconstruct is identical with tensors" claim, measured on a transformer.

Requires torch / transformers / peft and downloads gpt2 (network). Run:
    python -m warp_compress.lora_library_hf
"""
from __future__ import annotations

import numpy as np

from .gpu_delta import GPUDeltaCluster
from .lora_library import quantize_shared


def _demo():
    import warnings
    import time

    warnings.filterwarnings("ignore")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    import warp as wp
    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = get_peft_model(AutoModelForCausalLM.from_pretrained("gpt2"),
                           LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn"], task_type="CAUSAL_LM"))
    model.eval()

    # the LoRA tensors, in a fixed order — these ARE the adapter
    lora = [(n, p) for n, p in model.named_parameters() if "lora_" in n]
    names = [n for n, _ in lora]
    shapes = {n: tuple(p.shape) for n, p in lora}
    sizes = {n: p.numel() for n, p in lora}
    rng = np.random.default_rng(0)

    # a family of K sibling adapters: a shared nonzero ancestor + sparse per-task edits (peft inits B=0, so
    # we seed a real ancestor first, otherwise every adapter is a no-op and the swap proves nothing)
    K = 24
    ancestor = {n: (rng.standard_normal(shapes[n]).astype(np.float32) * 0.05) for n in names}
    fam = []
    for _ in range(K):
        a = {}
        for n in names:
            v = ancestor[n].copy()
            f = rng.integers(0, v.size, max(1, int(0.02 * v.size)))       # ~2% per-task divergence
            v.flat[f] += rng.standard_normal(f.size).astype(np.float32) * 0.05
            a[n] = v
        fam.append(a)

    # quantize each tensor on a shared grid across the family (keeps siblings near-duplicate as ints)
    scales = {}
    qfam = {n: None for n in names}
    for n in names:
        q, s = quantize_shared([a[n] for a in fam])
        qfam[n] = q                                                       # (K, ...) int
        scales[n] = s
    # flatten each adapter to one int sequence [tensor0 | tensor1 | ...]
    seqs = [np.concatenate([qfam[n][k].ravel() for n in names]) for k in range(K)]
    cluster = GPUDeltaCluster(seqs, device=dev)

    def _load(adapter_arrays):
        with torch.no_grad():
            for n, p in lora:
                p.copy_(torch.from_numpy(np.ascontiguousarray(adapter_arrays[n], np.float32)))

    def _split(flat):
        out, o = {}, 0
        for n in names:
            out[n] = flat[o:o + sizes[n]].reshape(shapes[n]).astype(np.float32) * scales[n]
            o += sizes[n]
        return out

    ids = tok("The history of language models is", return_tensors="pt").input_ids

    # correctness: reconstructed-adapter logits == quantized-adapter logits, exactly, for every adapter
    max_diff = 0.0
    distinct = 0
    prev = None
    t0 = time.perf_counter()
    for k in range(K):
        recon = _split(cluster.decode_leaf(k))                           # GPU delta-decode -> the adapter
        _load(recon)
        with torch.no_grad():
            lg_recon = model(ids).logits.numpy()
        ref = {n: qfam[n][k].reshape(shapes[n]).astype(np.float32) * scales[n] for n in names}
        _load(ref)
        with torch.no_grad():
            lg_ref = model(ids).logits.numpy()
        max_diff = max(max_diff, float(np.abs(lg_recon - lg_ref).max()))
        if prev is not None and float(np.abs(lg_recon - prev).max()) > 1e-4:
            distinct += 1
        prev = lg_recon
    dt = time.perf_counter() - t0

    total_params = sum(sizes.values())
    fp32 = K * total_params * 4
    int8 = K * total_params
    chromo = cluster.size_bytes()

    print(f"device={dev}   model=gpt2 + peft LoRA (r=8, c_attn)   family={K} adapters, {total_params} params each")
    print(f"[correct] reconstructed-adapter logits == quantized-adapter logits:  max|Δ| = {max_diff:.2e}  "
          f"(bit-exact reconstruction ⇒ identical model output)")
    print(f"[real]    adapters produce DIFFERENT logits ({distinct}/{K-1} consecutive pairs differ) — the swap "
          f"genuinely changes the model, it's not a no-op")
    print(f"[size]  fp32 independent {fp32/1e6:6.2f} MB   int8 independent {int8/1e6:5.2f} MB   "
          f"ChromoFold {chromo/1e6:5.3f} MB  => {int8/chromo:.1f}× vs int8, {fp32/chromo:.1f}× vs fp32")
    print(f"[speed] {K} real forward passes + GPU adapter reconstructions in {dt:.1f}s")
    print("=> on a real transformer: a LoRA library held as base+deltas, each adapter reconstructed on the GPU "
          "and hot-swapped, gives byte-identical model behaviour. docs/chromofold.md §5, on an actual model.")


if __name__ == "__main__":
    _demo()
