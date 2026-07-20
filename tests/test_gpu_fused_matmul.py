"""Fused decode-in-GEMM: weights decoded inside the matmul, matching decode-then-dense, without materialising W."""
import numpy as np

import warp as wp

from warp_compress.gpu_fused_matmul import FusedDecodeMatmul

_DEV = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"


def _rand(M, K, seed=0):
    return (np.random.default_rng(seed).standard_normal((M, K)) / np.sqrt(K)).astype(np.float32)


def test_fused_matches_decode_then_dense():
    W = _rand(256, 512, seed=1)
    x = _rand(4, 512, seed=2)
    fm = FusedDecodeMatmul(W, bits=4, block=64, device=_DEV)
    assert np.allclose(fm.matmul(x), fm.reference(x), atol=1e-4)   # same result, lossless over the quantization


def test_fused_never_needs_the_dense_matrix():
    W = _rand(512, 512, seed=3)
    fm = FusedDecodeMatmul(W, bits=4, block=64, device=_DEV)
    assert fm.resident_bytes() < fm.dense_bytes()                 # compressed store < dequantized fp32 matrix


def test_various_batch_and_block_sizes():
    W = _rand(300, 200, seed=4)
    for B in (1, 3, 16):
        x = _rand(B, 200, seed=10 + B)
        for blk in (16, 64, 128):
            fm = FusedDecodeMatmul(W, bits=4, block=blk, device=_DEV)
            assert np.allclose(fm.matmul(x), fm.reference(x), atol=1e-4)


def test_row_start_offset_is_correct_when_rows_span_blocks():
    # K not a multiple of block -> most rows start mid-block; the fused kernel must seek to each row's bit offset
    W = _rand(129, 100, seed=5)
    x = _rand(2, 100, seed=6)
    fm = FusedDecodeMatmul(W, bits=4, block=64, device=_DEV)
    assert np.allclose(fm.matmul(x), fm.reference(x), atol=1e-4)
