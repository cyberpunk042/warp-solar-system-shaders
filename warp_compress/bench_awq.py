"""bench_awq — AWQ calibration on a REAL gpt2: does activation-aware scaling make int4 usable? (lever #5 finish)

The capstone (`model_store`) found int4 degrades gpt2 (coherent at int8, "commas" at int4) — that's the
*quantizer's* cost, which ChromoFold's lossless entropy layer can't fix. AWQ is the fix: scale salient input
channels before quantizing. This measures it end-to-end on gpt2 — perplexity + a generation sample — for fp32
vs plain-int4 vs **AWQ-int4** vs int8, every quantized variant entropy-coded by ChromoFold (so the reported
size is the compressed footprint, and the weights stay GPU-addressable).

Requires torch/transformers + a cached gpt2. Run: python -m warp_compress.bench_awq
"""
from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore")

# a neutral expository passage — calibration (first half) + held-out perplexity (second half)
_TEXT = (
    "The history of science is the study of how human understanding of the natural world has developed over "
    "time. Early civilizations observed the motions of the stars and planets, recording their patterns to "
    "predict the seasons and guide agriculture. The ancient Greeks introduced systematic reasoning, proposing "
    "that the world could be understood through observation and logic rather than myth alone. During the "
    "medieval period, scholars in the Islamic world preserved and extended this knowledge, making advances in "
    "mathematics, astronomy, and medicine. The scientific revolution of the sixteenth and seventeenth centuries "
    "transformed these traditions into a rigorous method built on experiment and measurement. Figures such as "
    "Galileo and Newton demonstrated that the same laws governed motion on Earth and in the heavens, uniting "
    "physics under a small set of mathematical principles. In the following centuries, chemistry, biology, and "
    "geology matured into distinct disciplines, each with its own instruments and theories. The twentieth "
    "century brought relativity and quantum mechanics, which revealed that space, time, and matter behave in "
    "ways that defy everyday intuition. Today science advances through collaboration across many fields, and "
    "its methods continue to shape technology, medicine, and our understanding of ourselves."
)


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.pytorch_utils import Conv1D
    import warp as wp

    from .awq import awq_scale
    from .weight_store import QuantizedWeightStore

    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    tok = AutoTokenizer.from_pretrained("gpt2")
    ids = tok(_TEXT, return_tensors="pt").input_ids
    half = ids.shape[1] // 2
    cal_ids, eval_ids = ids[:, :half], ids[:, half:]
    targets = ("c_attn", "c_fc", "c_proj")                      # the transformer-block Conv1D weights

    # --- calibration: per-input-channel mean|x| for every target Conv1D (one forward pass) ---
    base = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    act_scale, hooks = {}, []
    for name, m in base.named_modules():
        if isinstance(m, Conv1D) and name.rsplit(".", 1)[-1] in targets:
            def hook(mod, inp, _out, nm=name):
                x = inp[0].detach().abs().reshape(-1, inp[0].shape[-1]).mean(0).numpy()
                act_scale[nm] = act_scale.get(nm, 0.0) + x
            hooks.append(m.register_forward_hook(hook))
    with torch.no_grad():
        base(cal_ids)
    for h in hooks:
        h.remove()

    def ppl(model):
        with torch.no_grad():
            return float(torch.exp(model(eval_ids, labels=eval_ids).loss))

    def build(bits, use_awq, group_size=None):
        model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
        total = 0
        for name, m in model.named_modules():
            if isinstance(m, Conv1D) and name.rsplit(".", 1)[-1] in targets:
                Wc = m.weight.detach().numpy().astype(np.float32)   # Conv1D weight is (in, out)
                Wlin = Wc.T                                         # -> (out, in): input channel = column
                cs = awq_scale(Wlin, act_scale[name], bits=bits, group_size=group_size)[0] if use_awq else None
                st = QuantizedWeightStore(Wlin, bits=bits, device=dev, huffman=True,
                                          group_size=group_size, channel_scale=cs)
                total += st.size_bytes()
                with torch.no_grad():
                    m.weight.copy_(torch.from_numpy(st.reconstruct().T.copy()))   # back to (in, out)
        return model, total

    fp32_ppl = ppl(base)
    gen_prompt = tok("The discovery of", return_tensors="pt").input_ids
    print(f"device={dev}   gpt2, quantizing {targets} across all blocks, entropy-coded by ChromoFold\n")
    print(f"  {'variant':>22} {'eval PPL':>9} {'vs fp32':>8} {'ChromoFold MB':>14}   sample")
    print(f"  {'fp32 (reference)':>22} {fp32_ppl:>9.2f} {'1.00×':>8} {'—':>14}")
    for label, bits, awq, gs in [("int8 plain", 8, False, None), ("int4 plain", 4, False, None),
                                 ("int4 group-128", 4, False, 128), ("int4 + AWQ", 4, True, None),
                                 ("int4 + AWQ g128", 4, True, 128)]:
        model, size = build(bits, awq, gs)
        p = ppl(model)
        with torch.no_grad():
            g = model.generate(gen_prompt, max_new_tokens=12, do_sample=False, pad_token_id=tok.eos_token_id)
        sample = tok.decode(g[0], skip_special_tokens=True).replace("\n", " ")
        print(f"  {label:>22} {p:>9.2f} {p/fp32_ppl:>7.2f}× {size/1e6:>13.2f}   {sample!r}")
    print("\n=> HONEST FINDING (gpt2): the lever that makes int4 usable is GROUP-WISE scaling — per-tensor int4 is "
          "broken (PPL ~4000, commas), group-128 is near-fp32 and coherent. AWQ's per-CHANNEL scaling did NOT\n"
          "   help here: per-tensor+AWQ stays broken (scaling salient channels up blows the single tensor scale), "
          "and group-128+AWQ ≈ group-128 alone. So on gpt2 the win is grouping (already a ChromoFold knob), not\n"
          "   AWQ. The `channel_scale` mechanism is built, lossless, and RA-preserving — it just isn't the lever "
          "for this model. ChromoFold entropy-codes whichever quantized form you pick and keeps it GPU-addressable.")


if __name__ == "__main__":
    main()
