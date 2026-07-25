"""Residual VQ: stacked codebooks refine the residual; addressable, lossless over the fp16 codebooks."""
import numpy as np

import warp as wp

from warp_compress.rvq_store import ResidualVQWeightStore
from warp_compress.vq_store import ProductQuantizedWeightStore

_DEV = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"


def _correlated(shape=(1024, 256), rank=8, seed=0):
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((shape[0], rank)).astype(np.float32)
    V = rng.standard_normal((rank, shape[1])).astype(np.float32)
    return ((U @ V) / np.sqrt(rank * shape[1]) + rng.standard_normal(shape) / np.sqrt(shape[1])).astype(np.float32)


def test_fetch_matches_reconstruct():
    W = _correlated(seed=1)
    st = ResidualVQWeightStore(W, subdim=4, codebook_bits=4, stages=3, device=_DEV)
    R = st.reconstruct().ravel()
    idx = np.random.default_rng(2).integers(0, st.n, 500)
    assert np.allclose(st.fetch(idx), R[idx], atol=1e-6)         # addressable == full decode (lossless/codebooks)


def test_reconstruct_is_additive_over_stages():
    W = _correlated((256, 128), seed=3)
    st = ResidualVQWeightStore(W, subdim=4, codebook_bits=4, stages=2, device=_DEV)
    idx = np.arange(st.n_codes, dtype=np.int64)
    ref = sum(cb[wm.access(idx)] for cb, wm in zip(st.codebooks, st.wms)).reshape(W.shape)
    assert np.array_equal(st.reconstruct(), ref.astype(np.float32))


def test_more_stages_lower_error():
    W = _correlated((1024, 256), seed=4)
    errs = [float(np.mean((ResidualVQWeightStore(W, subdim=4, codebook_bits=4, stages=s, device=_DEV)
                           .reconstruct() - W) ** 2)) for s in (1, 2, 3)]
    assert errs[0] > errs[1] > errs[2]                           # each stage refines the residual -> lower MSE


def test_bits_scale_with_stages():
    W = _correlated((1024, 256), seed=5)
    b1 = ResidualVQWeightStore(W, subdim=4, codebook_bits=4, stages=1, device=_DEV).bits_per_weight()
    b3 = ResidualVQWeightStore(W, subdim=4, codebook_bits=4, stages=3, device=_DEV).bits_per_weight()
    assert b3 > b1                                               # more stages cost more bits
    assert b3 < 3 * b1 + 0.5                                     # ~linear in stages (codebooks add a little)


def test_tiny_codebook_advantage():
    # RVQ reaches S*b code-bits with S*2**b centroids, vs 2**(S*b) for the single-codebook equivalent
    W = _correlated((512, 128), seed=6)
    S, b = 4, 4
    st = ResidualVQWeightStore(W, subdim=4, codebook_bits=b, stages=S, device=_DEV)
    total_centroids = sum(cb.shape[0] for cb in st.codebooks)    # S * 2**b = 64
    single_equiv = 1 << (S * b)                                  # 2**16 = 65536
    assert total_centroids <= S * (1 << b)
    assert total_centroids * 50 < single_equiv                  # orders of magnitude fewer centroids


def test_competitive_with_single_codebook_at_matched_bits():
    # honest: at matched code-bits a single free codebook is a bit BETTER; RVQ stays within a small factor
    W = _correlated((2048, 512), rank=8, seed=7)
    single = ProductQuantizedWeightStore(W, subdim=4, codebook_bits=8, device=_DEV)   # 8 code-bits
    rvq = ResidualVQWeightStore(W, subdim=4, codebook_bits=4, stages=2, device=_DEV)  # 2x4 = 8 code-bits
    assert abs(rvq.bits_per_weight() - single.bits_per_weight()) < 0.4
    mse = lambda st: float(np.mean((st.reconstruct() - W) ** 2))
    assert mse(rvq) < 1.6 * mse(single)                         # competitive (a bit worse — additive constraint)


def test_size_must_be_multiple_of_subdim():
    W = np.zeros((10, 7), np.float32)
    try:
        ResidualVQWeightStore(W, subdim=4, codebook_bits=4, stages=2, device=_DEV)
        assert False, "expected ValueError"
    except ValueError:
        pass
