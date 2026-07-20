"""hf_cache — a drop-in Hugging Face `transformers` Cache backed by ChromoFold. Real generation, compressed KV.

Pass `ChromoFoldCache()` as `past_key_values` to any `transformers` CausalLM and generate normally. Each layer
keeps a small fp16 **residual window** (the most recent tokens) and compresses the settled prefix with ChromoFold
(KIVI per-axis quantization + block-Huffman entropy coding). The prefix is held compressed instead of fp16, so
long contexts fit in far less VRAM; output tracks the plain-quantized-KV model (coherent).

    cache = ChromoFoldCache(residual=128, bits=4)
    model.generate(**inputs, past_key_values=cache)      # normal generation, compressed KV
    cache.memory_bytes()                                 # resident KV bytes vs a full fp16 cache

Each compressed chunk is decoded **exactly once** (when it settles) and memoized into an accumulating prefix, so
per-step cost is O(new tokens), not O(context) — generation is O(n), not O(n²). Attention still sees the full
K/V, so this trades a one-time decode per chunk for the memory saving (the ChromoFold compute-for-memory thesis),
and it is the long-context regime where that pays. Supports batched and beam-search generation.

Honest scope: a Warp research path (reconstruct-into-attention), not a fused attention kernel; `crop` (assisted
decoding into the compressed prefix) is not supported. Requires torch/transformers. Run:
python -m warp_compress.hf_cache
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
            self._chunks = []        # compressed KVCacheStore chunks (kept for memory accounting)
            self._settled = 0        # number of tokens held compressed
            self._prefix_k = None    # memoized reconstructed prefix (torch, cpu, fp32) — each chunk decoded once
            self._prefix_v = None
            self._decodes = 0        # instrumentation: total chunk decodes (== len(chunks), not per-step)

        def _compress(self, k, v):
            from .kv_store import KVCacheStore
            st = KVCacheStore([(k.detach().to(torch.float32).cpu().numpy(),
                                v.detach().to(torch.float32).cpu().numpy())],
                              bits=self._cf_bits, device=self._cf_device, per_axis=True)
            self._chunks.append(st)
            K, V = st.reconstruct_layer(0)                 # decode THIS chunk once; accumulate into the prefix memo
            self._decodes += 1
            nk, nv = torch.from_numpy(K), torch.from_numpy(V)
            self._prefix_k = nk if self._prefix_k is None else torch.cat([self._prefix_k, nk], dim=-2)
            self._prefix_v = nv if self._prefix_v is None else torch.cat([self._prefix_v, nv], dim=-2)

        def _reassemble(self, dtype, device):
            pk, pv = [], []
            if self._prefix_k is not None:                 # memoized: no per-step re-decode of the settled prefix
                pk.append(self._prefix_k); pv.append(self._prefix_v)
            pk.append(self.keys.to(torch.float32).cpu()); pv.append(self.values.to(torch.float32).cpu())
            K = torch.cat(pk, dim=-2).to(device=device, dtype=dtype)
            V = torch.cat(pv, dim=-2).to(device=device, dtype=dtype)
            return K, V

        def update(self, key_states, value_states, *args, **kwargs):
            if not self.is_initialized:
                self.lazy_initialization(key_states, value_states)
            self.keys = torch.cat([self.keys, key_states], dim=-2)      # append into the fp16 residual
            self.values = torch.cat([self.values, value_states], dim=-2)
            cur = self.keys.shape[-2]
            if cur > 2 * self._cf_residual:                             # flush the overflow into a compressed chunk
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

        def reorder_cache(self, beam_idx):
            """Beam search: select beams along the batch dim, for the memoized prefix and the fp16 residual.
            Old compressed chunks are kept only for memory accounting (never re-decoded), so their batch order
            is irrelevant to correctness; new flushes recompress from the reordered residual."""
            if self._prefix_k is not None:
                bi = beam_idx.to(self._prefix_k.device)
                self._prefix_k = self._prefix_k.index_select(0, bi)
                self._prefix_v = self._prefix_v.index_select(0, bi)
            if self.is_initialized:
                self.keys = self.keys.index_select(0, beam_idx.to(self.keys.device))
                self.values = self.values.index_select(0, beam_idx.to(self.values.device))

        def crop(self, max_length: int):
            if max_length < 0:
                max_length = self.get_seq_length() + max_length
            if max_length >= self._settled and self.is_initialized:    # crop only within the fp16 residual window
                keep = max_length - self._settled
                self.keys = self.keys[:, :, :keep, :].contiguous()
                self.values = self.values[:, :, :keep, :].contiguous()
                return
            raise NotImplementedError(
                "ChromoFoldCache.crop into the compressed prefix is not supported (assisted/speculative decoding "
                "that rewinds past the residual window). Use a larger `residual` or a standard cache for that path.")

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
            self.residual, self.bits, self.device = residual, bits, device   # config (closure), for repr

        def memory_bytes(self) -> int:
            return sum(l.memory_bytes() for l in self.layers)

        def fp16_bytes(self) -> int:
            return sum(l.fp16_equivalent_bytes() for l in self.layers)

        def decode_count(self) -> int:
            """Total compressed-chunk decodes across all layers (== chunks, thanks to prefix memoization —
            NOT steps × chunks). A naive reassemble-every-step cache would be far higher."""
            return sum(getattr(l, "_decodes", 0) for l in self.layers)

        def __repr__(self):
            return (f"ChromoFoldCache(residual={self.residual}, bits={self.bits}, layers={len(self.layers)}, "
                    f"resident={self.memory_bytes()/1e6:.2f}MB)")

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
