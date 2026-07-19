"""GPU (Warp) RRR bitvector: batched rank1 is exact, and skewed vectors compress below packed."""
import numpy as np

import warp as wp

from warp_compress.gpu_rrr import GPURRR

_DEV = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"


def _check_rank(bits, seed=0):
    gr = GPURRR(bits, device=_DEV)
    cum = np.concatenate([[0], np.cumsum(np.asarray(bits) != 0)])
    rng = np.random.default_rng(seed)
    q = rng.integers(0, len(bits) + 1, 3000).astype(np.int32)
    got = gr.rank1(q)
    assert np.array_equal(got, cum[q])
    return gr


def test_rank1_exact_on_skewed():
    rng = np.random.default_rng(1)
    _check_rank((rng.random(60000) < 0.03).astype(np.uint8))


def test_rank1_exact_on_balanced():
    rng = np.random.default_rng(2)
    _check_rank((rng.random(60000) < 0.5).astype(np.uint8))


def test_rank1_exact_on_all_zero_and_all_one():
    _check_rank(np.zeros(5000, np.uint8))
    _check_rank(np.ones(5000, np.uint8))


def test_rank1_handles_boundaries():
    rng = np.random.default_rng(3)
    bits = (rng.random(4000) < 0.2).astype(np.uint8)
    gr = GPURRR(bits, device=_DEV)
    cum = np.concatenate([[0], np.cumsum(bits)])
    q = np.array([0, 1, 14, 15, 16, len(bits) - 1, len(bits)], np.int32)   # block edges + ends
    assert np.array_equal(gr.rank1(q), cum[q])


def test_skewed_compresses_below_packed():
    rng = np.random.default_rng(4)
    gr = GPURRR((rng.random(200000) < 0.02).astype(np.uint8), device=_DEV)
    assert gr.size_bits() < gr.n            # a very skewed plane costs < 1 bit/bit (packed = n bits)
