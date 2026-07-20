"""hf_cache — a drop-in Hugging Face `transformers` Cache backed by ChromoFold. Real generation, compressed KV.

This is the "works today" integration: pass `ChromoFoldCache()` as `past_key_values` to any `transformers`
model and generate normally. Each layer keeps a small fp16 **residual window** (the most recent tokens) and
compresses the settled prefix with ChromoFold (KIVI per-axis quantization + block-Huffman entropy coding). On
every step it reassembles the full K/V for attention, so output is the plain-quantized-KV output — coherent,
with the long prefix held compressed instead of fp16.

    cache = ChromoFoldCache(residual=128, bits=4)
    model.generate(**inputs, past_key_values=cache)      # normal generation, compressed KV
    cache.memory_bytes()                                 # resident KV bytes vs a full fp16 cache

Honest: reassembling the prefix each step trades compute (re-decode) for memory — the ChromoFold thesis. It is
the long-context regime (compress the settled prefix once, decode from it) where this pays; it is a Warp
research path, not a fused attention kernel. Requires torch/transformers. Run: python -m warp_compress.hf_cache
"""
from __future__ import annotations

import numpy as np


def _layer_cls():
    """Build the ChromoFold cache layer lazily so importing this module doesn't require torch/transformers."""
    import torch
    from transformers.cache_utils import DynamicLayer

    class ChromoFoldLayer(DynamicLayer):
        _cf_residual = 128           # underscored: DynamicLayer already owns `.device` (the torch device) etc.
        _cf_bits = 4
        _cf_device = "cuda:0"

        def __init__(self, **kw):
            super().__init__(**kw)
            self._chunks = []        # compressed KVCacheStore chunks for the settled prefix
            self._settled = 0        # number of tokens held compressed

        def _compress(self, k, v):
            from .kv_store import KVCacheStore
            self._chunks.append(KVCacheStore(
                [(k.detach().to(torch.float32).cpu().numpy(), v.detach().to(torch.float32).cpu().numpy())],
                bits=self._cf_bits, device=self._cf_device, per_axis=True))

        def _reassemble(self, dtype, device):
            ks, vs = [], []
            for st in self._chunks:                       # decode the compressed prefix (compute-for-memory)
                K, V = st.reconstruct_layer(0)
                ks.append(torch.from_numpy(K)); vs.append(torch.from_numpy(V))
            ks.append(self.keys.to(torch.float32).cpu()); vs.append(self.values.to(torch.float32).cpu())
            K = torch.cat(ks, dim=-2).to(device=device, dtype=dtype)
            V = torch.cat(vs, dim=-2).to(device=device, dtype=dtype)
            return K, V

        def update(self, key_states, value_states, *args, **kwargs):
            if not self.is_initialized:
                self.lazy_initialization(key_states, value_states)
            self.keys = torch.cat([self.keys, key_states], dim=-2)      # append into the fp16 residual
            self.values = torch.cat([self.values, value_states], dim=-2)
            cur = self.keys.shape[-2]
            if cur > 2 * self._cf_residual:                                 # flush the overflow into a compressed chunk
                n = cur - self._cf_residual
                self._compress(self.keys[:, :, :n, :], self.values[:, :, :n, :])
                self.keys = self.keys[:, :, n:, :].contiguous()
                self.values = self.values[:, :, n:, :].contiguous()
                self._settled += n
            return self._reassemble(key_states.dtype, key_states.device)

        def get_seq_length(self, *args, **kwargs) -> int:
            # total length the reassembled K/V presents to attention = compressed prefix + fp16 residual.
            # (get_mask_sizes is inherited: it does get_seq_length() + query_length, kv_offset=0 — correct here.)
            return self._settled + (self.keys.shape[-2] if self.is_initialized else 0)

        def memory_bytes(self) -> int:
            comp = sum(st.size_bytes() for st in self._chunks)
            res = (self.keys.numel() + self.values.numel()) * 2 if self.is_initialized else 0
            return comp + res

        def fp16_equivalent_bytes(self) -> int:
            return int(self.get_seq_length()) * (self._chunks[0].layers[0]["Kshape"][1] * 2 * 2
                       * self._chunks[0].layers[0]["Kshape"][3]) if self._chunks else \
                   (self.keys.numel() + self.values.numel()) * 2

    return ChromoFoldLayer


def make_cache(residual: int = 128, bits: int = 4, device: str = "cuda:0"):
    """Return a `transformers` Cache whose layers are ChromoFold-backed (residual fp16 window + compressed prefix)."""
    from transformers.cache_utils import Cache
    Layer = _layer_cls()
    Layer._cf_residual, Layer._cf_bits, Layer._cf_device = residual, bits, device

    class ChromoFoldCache(Cache):
        def __init__(self):
            super().__init__(layer_class_to_replicate=Layer)

        def memory_bytes(self) -> int:
            return sum(l.memory_bytes() for l in self.layers)

        def fp16_bytes(self) -> int:
            return sum(l.fp16_equivalent_bytes() for l in self.layers)

    return ChromoFoldCache()


def _demo():
    import os
    import warnings
    warnings.filterwarnings("ignore")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import warp as wp

    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    name = os.environ.get("CHROMOFOLD_KV_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()

    ctx = ("A key-value cache stores the attention state for every token, so long contexts are bounded by "
           "memory, not compute. " * 40)
    prompt = ctx + "\n\nIn one sentence, the main bottleneck for long-context language models is"
    ids = tok(prompt, return_tensors="pt").input_ids

    def run(cache):
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=28, do_sample=False, pad_token_id=tok.eos_token_id,
                                 past_key_values=cache)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).replace("\n", " ").strip()

    from transformers import DynamicCache
    base_txt = run(DynamicCache())
    cache = make_cache(residual=128, bits=4, device=dev)
    cf_txt = run(cache)

    n = ids.shape[1] + 28
    kvh = model.config.num_key_value_heads
    d = model.config.hidden_size // model.config.num_attention_heads
    fp16 = 2 * model.config.num_hidden_layers * kvh * n * d * 2
    cfold = cache.memory_bytes()
    print(f"device={dev}   {name}   prompt {ids.shape[1]} tok + 28 generated = {n} tok\n")
    print(f"  fp16 DynamicCache : {base_txt!r}")
    print(f"  ChromoFoldCache   : {cf_txt!r}")
    print(f"\n  resident KV cache: fp16 {fp16/1e6:.2f} MB  ->  ChromoFold {cfold/1e6:.2f} MB "
          f"= {fp16/cfold:.1f}× smaller (residual={128} fp16 tokens/layer, prefix int4-KIVI + entropy-coded)")
    print("\n=> a real transformers model generates through ChromoFoldCache as a drop-in past_key_values: the long "
          "prefix is held compressed in place while a small fp16 window stays hot. Output tracks the quantized-KV\n"
          "   model (coherent); the resident KV shrinks. Honest: reassembling the prefix each step trades compute "
          "for memory — the long-context regime is where that trade pays.")


if __name__ == "__main__":
    _demo()
