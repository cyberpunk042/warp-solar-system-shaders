"""Per-subspace product quantization: a codebook per column-block; addressable, lossless over the codebooks."""
import numpy as np

import warp as wp

from warp_compress.pq_subspace import SubspacePQWeightStore
from warp_compress.vq_store import ProductQuantizedWeightStore

_DEV = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"


def _heterogeneous(shape=(1024, 256), rank=8, seed=0):
    # columns with different scales -> exactly what a per-subspace codebook exploits
    rng = np.random.default_rng(seed)
    base = (rng.standard_normal((shape[0], rank)) @ rng.standard_normal((rank, shape[1]))) / np.sqrt(rank * shape[1])
    cs = (0.2 + 2.0 * rng.random(shape[1])).astype(np.float32)
    return (base * cs[None, :] + rng.standard_normal(shape) / np.sqrt(shape[1])).astype(np.float32)


def test_fetch_matches_reconstruct():
    W = _heterogeneous(seed=1)
    st = SubspacePQWeightStore(W, subdim=4, codebook_bits=8, device=_DEV)
    R = st.reconstruct().ravel()
    idx = np.random.default_rng(2).integers(0, st.n, 500)
    assert np.allclose(st.fetch(idx), R[idx], atol=1e-6)         # addressable == full decode (lossless/codebooks)


def test_reconstruct_is_per_block_centroid_gather():
    W = _heterogeneous((256, 128), seed=3)
    st = SubspacePQWeightStore(W, subdim=4, codebook_bits=6, device=_DEV)
    ref = np.empty((st.out, st.n_sub, st.subdim), np.float32)
    idx = np.arange(st.out, dtype=np.int64)
    for s, (cb, wm) in enumerate(zip(st.codebooks, st.wms)):
        ref[:, s, :] = cb[wm.access(idx)]
    assert np.array_equal(st.reconstruct(), ref.reshape(W.shape))


def test_fetch_rows_matches_reconstruct():
    W = _heterogeneous((320, 128), seed=4)
    st = SubspacePQWeightStore(W, subdim=4, codebook_bits=8, device=_DEV)
    R = st.reconstruct()
    rows = np.array([0, 7, 100, 319])
    assert np.allclose(st.fetch_rows(rows), R[rows], atol=1e-6)


def test_lower_error_than_shared_at_same_code_width():
    # the fit advantage: at the SAME subdim + codebook_bits, per-subspace codebooks give lower reconstruction MSE
    W = _heterogeneous((1024, 256), seed=5)
    shared = ProductQuantizedWeightStore(W, subdim=4, codebook_bits=8, device=_DEV)
    sub = SubspacePQWeightStore(W, subdim=4, codebook_bits=8, device=_DEV)
    mse = lambda st: float(np.mean((st.reconstruct() - W) ** 2))
    assert mse(sub) < mse(shared)                                # per-block fit beats one global codebook


def test_costs_more_bits_from_codebook_tax():
    # honest: per-subspace stores in/subdim codebooks -> ~k/out extra bits/weight (real on a narrow tensor)
    W = _heterogeneous((1024, 256), seed=6)
    shared = ProductQuantizedWeightStore(W, subdim=4, codebook_bits=8, device=_DEV)
    sub = SubspacePQWeightStore(W, subdim=4, codebook_bits=8, device=_DEV)
    assert sub.bits_per_weight() > shared.bits_per_weight()      # the per-subspace codebook overhead
    assert sub.n_sub == W.shape[1] // 4 and len(sub.codebooks) == sub.n_sub


def test_in_features_must_be_multiple_of_subdim():
    W = np.zeros((16, 10), np.float32)                          # 10 not divisible by 4
    try:
        SubspacePQWeightStore(W, subdim=4, codebook_bits=6, device=_DEV)
        assert False, "expected ValueError"
    except ValueError:
        pass
