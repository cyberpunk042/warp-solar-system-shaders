"""PQ decode-from-codebook GEMM: y = x·Ŵᵀ with weights decoded in-kernel; no dense W materialized."""
import numpy as np

import warp as wp

from warp_compress.pq_matmul import PQDecodeMatmul

_DEV = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"


def _weights(M=256, K=128, rank=8, seed=0):
    rng = np.random.default_rng(seed)
    return ((rng.standard_normal((M, rank)) @ rng.standard_normal((rank, K))) / np.sqrt(rank * K)
            + rng.standard_normal((M, K)) / np.sqrt(K)).astype(np.float32)


def test_matches_decode_then_gemm():
    W = _weights(seed=1)
    x = np.random.default_rng(2).standard_normal((8, W.shape[1])).astype(np.float32)
    mm = PQDecodeMatmul(W, subdim=4, codebook_bits=8, device=_DEV)
    y = mm.matmul(x)
    y_ref = x @ mm.store.reconstruct().T                        # same PQ weights, decoded first
    assert y.shape == (8, W.shape[0])
    assert np.max(np.abs(y - y_ref)) / (np.abs(y_ref).max() + 1e-9) < 1e-5   # fused == decode-then-GEMM


def test_no_dense_weight_resident():
    W = _weights(512, 256, seed=3)
    mm = PQDecodeMatmul(W, subdim=4, codebook_bits=8, device=_DEV)
    assert mm.resident_bytes() < mm.dense_bytes()
    assert mm.dense_bytes() / mm.resident_bytes() > 10.0        # ~16x for 8b PQ subdim4 (2 bits/weight)


def test_resident_is_uint8_codes_plus_fp16_codebook():
    W = _weights(seed=4)
    mm = PQDecodeMatmul(W, subdim=4, codebook_bits=8, device=_DEV)
    expected = mm.store.n_codes * 1 + mm.store.codebook.size * 2
    assert mm.resident_bytes() == expected                     # exactly the compact PQ form, nothing else


def test_batch_and_shapes():
    W = _weights(300, 128, seed=5)
    mm = PQDecodeMatmul(W, subdim=4, codebook_bits=8, device=_DEV)
    for B in (1, 4, 32):
        x = np.random.default_rng(B).standard_normal((B, 128)).astype(np.float32)
        assert mm.matmul(x).shape == (B, 300)


def test_codebook_bits_over_8_rejected():
    W = _weights(64, 64, seed=6)
    try:
        PQDecodeMatmul(W, subdim=4, codebook_bits=10, device=_DEV)
        assert False, "expected ValueError (codes are packed as uint8)"
    except ValueError:
        pass
