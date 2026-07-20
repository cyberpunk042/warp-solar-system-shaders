"""Long-context generation with a ChromoFold-compressed KV cache — the sovereign / on-prem serving pattern.

Drops `ChromoFoldCache` in as `past_key_values`: the settled prefix is held compressed while a small fp16 window
stays hot, so a long context fits in far less VRAM. Runs fully offline (point transformers at local weights).

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python examples/serve_compressed_kv.py [model_name]

Requires `pip install chromofold[torch]` and a locally-available CausalLM (default Qwen2.5-0.5B-Instruct).
"""
import sys
import warnings

warnings.filterwarnings("ignore")


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    import chromofold as cf

    name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()

    context = ("A key-value cache stores the attention state for every token, so long contexts are bounded by "
               "memory, not compute. " * 60)
    prompt = context + "\n\nIn one sentence, the main bottleneck for long-context language models is"
    ids = tok(prompt, return_tensors="pt").input_ids

    def generate(cache):
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=24, do_sample=False, pad_token_id=tok.eos_token_id,
                                 past_key_values=cache)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).replace("\n", " ").strip()

    print(f"model {name}  ·  prompt {ids.shape[1]} tokens + 24 generated\n")
    print("fp16 KV     :", repr(generate(DynamicCache())))

    cache = cf.ChromoFoldCache(residual=128, bits=4)
    print("ChromoFold  :", repr(generate(cache)))
    print(f"\n{cache!r}")
    print(f"chunk decodes (== chunks, memoized, not steps x chunks): {cache.decode_count()}")
    print("=> the long prefix is held compressed; each chunk decoded once. Longer context -> larger VRAM win.")


if __name__ == "__main__":
    main()
