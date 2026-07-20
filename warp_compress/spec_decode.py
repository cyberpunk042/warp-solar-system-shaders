"""spec_decode — the ChromoFold self-index AS a speculative-decoding draft model, on a real model (gpt2).

Advantage #3 of `docs/chromofold_positioning.md`, made concrete: the same compressed FM-index that stores a
context is also a *predictor* over it. Use it as the cheap **draft** in speculative decoding — it proposes the
continuation of the longest recurring suffix (a suffix-index generalisation of prompt-lookup decoding) — and
let the big model **verify** a whole draft in one forward pass. The output is byte-identical to greedy decoding
(speculative decoding is exact), but the number of expensive model forwards drops whenever the text repeats
patterns the index has seen (RAG, code, structured output, any self-similar generation).

    draft(fm, context, k)                  -> up to k proposed tokens (continuation of the longest prior match)
    speculative_generate(model, tok, …)    -> (tokens, model_forwards, accepted) — same tokens as greedy

Requires torch/transformers. Run: python -m warp_compress.spec_decode
"""
from __future__ import annotations

import time
import warnings

import numpy as np

from .fm_index import FMIndex

warnings.filterwarnings("ignore")


def draft(fm: FMIndex, context, max_draft: int = 8, max_order: int = 12):
    """Propose the continuation of the LONGEST suffix of `context` that occurred earlier — via the FM-index
    (backward search for the suffix, locate a prior occurrence, take what followed it)."""
    ctx = list(context)
    n = len(ctx)
    for L in range(min(max_order, n), 0, -1):
        pat = ctx[n - L:]
        locs = fm.locate(pat)
        for p in locs:                                          # a prior occurrence with tokens after it
            if p + L < n:
                return ctx[p + L: p + L + max_draft]
    return []


def speculative_generate(model, tok, prompt_ids, n_new: int, max_draft: int = 8, rebuild_every: int = 1):
    """Greedy generation accelerated by the FM-index draft. Returns (new_tokens, model_forwards, n_generated)."""
    import torch
    ctx = [int(x) for x in prompt_ids]
    forwards = 0
    fm = FMIndex(np.asarray(ctx, np.int64))
    since = 0
    start = len(ctx)
    while len(ctx) - start < n_new:
        d = draft(fm, ctx, max_draft)
        inp = torch.tensor([ctx + d], dtype=torch.long)
        with torch.no_grad():
            logits = model(inp).logits[0]                       # (len(ctx)+len(d), vocab)
        forwards += 1
        base = len(ctx)
        # p[i] = the model's greedy token for the i-th slot after ctx (position base-1+i predicts slot i)
        p = [int(logits[base - 1 + i].argmax()) for i in range(len(d) + 1)]
        m = 0
        while m < len(d) and d[m] == p[m]:                      # accept drafts the model would have chosen
            m += 1
        accepted = p[: m + 1]                                   # m matched drafts + 1 bonus/correction token
        ctx.extend(accepted)
        since += len(accepted)
        if since >= rebuild_every:                              # refresh the index over the grown context
            fm = FMIndex(np.asarray(ctx, np.int64))
            since = 0
    new = ctx[start:start + n_new]
    return new, forwards, len(new)


def _greedy(model, prompt_ids, n_new):
    import torch
    ctx = [int(x) for x in prompt_ids]
    start = len(ctx)
    forwards = 0
    while len(ctx) - start < n_new:
        with torch.no_grad():
            nxt = int(model(torch.tensor([ctx], dtype=torch.long)).logits[0, -1].argmax())
        ctx.append(nxt)
        forwards += 1
    return ctx[start:start + n_new], forwards


def _demo():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()

    # a RAG-flavour prompt: a passage the model will lean on / repeat — where the index-draft pays off
    passage = ("ChromoFold is a GPU-resident, random-access, searchable entropy code for language-model data. "
               "ChromoFold stores weights, KV cache, experts, and context compressed in VRAM. ")
    prompt = passage + "In summary, ChromoFold is a"
    ids = tok(prompt, return_tensors="pt").input_ids[0]
    N = 48

    g_tokens, g_fwd = _greedy(model, ids, N)
    t0 = time.perf_counter()
    s_tokens, s_fwd, _ = speculative_generate(model, tok, ids, N, max_draft=8)
    s_t = time.perf_counter() - t0

    exact = g_tokens == s_tokens                                # speculative decoding is EXACT vs greedy
    print(f"model=gpt2   generate {N} tokens   prompt has recurring structure (RAG-flavour)")
    print(f"[correct] speculative output == greedy output: {exact}")
    print(f"[speed]   greedy: {g_fwd} model forwards ({N} tokens)   "
          f"index-draft speculative: {s_fwd} model forwards   => {g_fwd / s_fwd:.2f}× fewer big-model passes")
    print(f"          ({N / s_fwd:.2f} tokens accepted per model forward on average)")
    print(f"  continuation: {tok.decode(s_tokens, skip_special_tokens=True)[:120]!r}")
    print("\n=> the compressed FM-index IS the draft model: it proposes the continuation of the longest recurring "
          "suffix (search in the compressed domain), the big model verifies a whole draft in one pass, and the\n"
          "   output is identical to greedy. Fewer expensive forwards whenever generation is self-similar "
          "(RAG / code / structured) — advantage #3, on a real model. No extra model, no training.")


if __name__ == "__main__":
    _demo()
