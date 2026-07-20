"""chromofold — GPU-resident, random-access, *searchable* compression for LLM data.

ChromoFold keeps the data an LLM runs on — token streams, KV cache, quantized weights, MoE experts,
adapters, prompt caches — compressed **and navigable** in GPU memory: addressable in O(1)/O(log) and
searchable in the compressed domain, decoded only where consumed. It composes on top of quantization
(it is the lossless entropy + random-access layer, not the quantizer).

Public API (importing this package does **no** network I/O and does **not** require torch):

    import chromofold as cf
    art = cf.compress(weights_or_tokens)      # -> Artifact: .decode() / .fetch(idx) / .save() / .size_bytes()
    cf.QuantizedWeightStore(W, bits=4)        # quantized + entropy-coded weights, GPU-addressable
    cf.KVCacheStore(past_key_values)          # KIVI per-axis KV + entropy, windowed (attended-only) decode
    cf.ChromoFoldCache(residual=128, bits=4)  # drop-in transformers `past_key_values`  [needs: pip install chromofold[torch]]

The reference implementation is Python + NVIDIA Warp (offline, air-gap-friendly). A native C++/CUDA
engine with a stable C ABI is developed alongside it. See INTEGRATION.md for Hugging Face and
sovereign / on-prem usage.
"""
from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["compress", "Artifact", "QuantizedWeightStore", "KVCacheStore", "MoEExpertStore",
           "ChromoFoldCache", "__version__"]

# Lazy attribute map: the heavy modules (numpy/warp, and torch only for the cache) load on first use,
# so `import chromofold` stays fast, offline, and torch-free.
_LAZY = {
    "compress": ("warp_compress.api", "compress"),
    "Artifact": ("warp_compress.api", "Artifact"),
    "QuantizedWeightStore": ("warp_compress.weight_store", "QuantizedWeightStore"),
    "KVCacheStore": ("warp_compress.kv_store", "KVCacheStore"),
    "MoEExpertStore": ("warp_compress.moe_store", "MoEExpertStore"),
    "ChromoFoldCache": ("warp_compress.hf_cache", "make_cache"),  # factory: ChromoFoldCache(residual=, bits=)
}


def __getattr__(name):  # PEP 562 lazy loading
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'chromofold' has no attribute {name!r}")
    import importlib
    module, attr = target
    return getattr(importlib.import_module(module), attr)


def __dir__():
    return sorted(__all__)
