"""ChromoFold LoRA library: quantized adapter family stored as base+deltas, reconstructed bit-exact on GPU."""
import numpy as np

from warp_compress.lora_library import ChromoLoRALibrary, quantize_shared, synth_family

import warp as wp

_DEV = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"


def test_reconstruct_is_bit_exact_vs_quantized_adapter():
    _, As, Bs = synth_family(d_out=256, d_in=256, r=8, k=16, seed=1)
    lib = ChromoLoRALibrary(As, Bs, device=_DEV)
    for k in (0, 5, 15):
        A, B = lib.reconstruct(k)
        assert np.array_equal(A, lib.Aq[k].astype(np.float32) * lib.sa)
        assert np.array_equal(B, lib.Bq[k].astype(np.float32) * lib.sb)


def test_forward_pass_matches_reference():
    W, As, Bs = synth_family(d_out=256, d_in=256, r=8, k=12, seed=2)
    lib = ChromoLoRALibrary(As, Bs, device=_DEV)
    x = np.random.default_rng(3).standard_normal((4, 256)).astype(np.float32)
    for k in (1, 7, 11):
        A, B = lib.reconstruct(k)
        y = lib.apply(x, k, W)
        assert np.allclose(y, x @ (W + (B @ A)).T, atol=1e-4)


def test_library_smaller_than_independent_int8():
    _, As, Bs = synth_family(d_out=512, d_in=512, r=16, k=32, perturb=0.02, seed=4)
    lib = ChromoLoRALibrary(As, Bs, device=_DEV)
    assert lib.size_bytes() < lib._raw_int8      # base + sparse deltas beats storing every adapter's int8


def test_shared_quant_keeps_siblings_near_duplicate():
    _, As, _ = synth_family(d_out=128, d_in=128, r=8, k=8, perturb=0.02, seed=5)
    q, scale = quantize_shared(As)
    assert scale > 0
    diff = np.count_nonzero(q[1] != q[0]) / q[0].size
    assert diff < 0.15                            # a sibling differs from the ancestor in only a few entries
