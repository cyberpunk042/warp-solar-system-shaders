"""Product-quantized weight store: vector codebook + GPU-addressable codes, lossless over the codebook."""
import numpy as np

import warp as wp

from warp_compress.vq_store import ProductQuantizedWeightStore
from warp_compress.weight_store import QuantizedWeightStore

_DEV = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"


def _correlated(shape=(512, 256), rank=8, seed=0):
    # low-rank structure + noise -> correlated sub-vectors, the regime a vector codebook exploits
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((shape[0], rank)).astype(np.float32)
    V = rng.standard_normal((rank, shape[1])).astype(np.float32)
    return ((U @ V) / np.sqrt(rank * shape[1]) + rng.standard_normal(shape) / np.sqrt(shape[1])).astype(np.float32)


def test_fetch_matches_reconstruct():
    W = _correlated(seed=1)
    st = ProductQuantizedWeightStore(W, subdim=4, codebook_bits=8, device=_DEV)
    R = st.reconstruct().ravel()
    idx = np.random.default_rng(2).integers(0, st.n, 500)
    assert np.allclose(st.fetch(idx), R[idx], atol=1e-6)          # random access == full decode (lossless/codebook)


def test_reconstruct_is_exact_centroid_gather():
    # "lossless over the codebook": reconstruct must equal gathering the stored centroids by the stored codes
    W = _correlated((256, 128), seed=3)
    st = ProductQuantizedWeightStore(W, subdim=4, codebook_bits=6, device=_DEV)
    codes = st.wm.access(np.arange(st.n_codes, dtype=np.int64))
    ref = st.codebook[codes].reshape(W.shape)
    assert np.array_equal(st.reconstruct(), ref)


def test_fetch_rows_matches_reconstruct():
    W = _correlated((320, 128), seed=4)
    st = ProductQuantizedWeightStore(W, subdim=4, codebook_bits=8, device=_DEV)
    R = st.reconstruct()
    rows = np.array([0, 5, 17, 319])
    assert np.allclose(st.fetch_rows(rows), R[rows], atol=1e-6)


def test_bits_per_weight_near_codebook_rate():
    # the ratio is the vector codebook (log2(k)/subdim); balanced k-means => codes near-uniform, so the RRR
    # index is ~fixed-width and entropy coding adds little. b/weight sits near the ceiling + small overhead.
    W = _correlated((1024, 256), seed=5)
    st = ProductQuantizedWeightStore(W, subdim=4, codebook_bits=8, device=_DEV)
    raw_ceiling = st.codebook_bits / st.subdim                    # 8/4 = 2.0 b/weight
    assert st.bits_per_weight() < raw_ceiling + 0.6              # codebook + index overhead is bounded
    assert st.bits_per_weight() < st.codebook_bits              # still well under storing raw code ids


def test_pq_beats_scalar_int3_in_low_bit_regime():
    # matched ~1.7 b/weight: the vector codebook should give lower reconstruction error than scalar int3
    W = _correlated((2048, 512), rank=8, seed=6)
    int3 = QuantizedWeightStore(W, bits=3, huffman=True, device=_DEV)
    pq = ProductQuantizedWeightStore(W, subdim=4, codebook_bits=6, device=_DEV)
    assert abs(pq.bits_per_weight() - int3.bits_per_weight()) < 0.4    # comparable rate
    mse = lambda R: float(np.mean((R - W) ** 2))
    assert mse(pq.reconstruct()) < mse(int3.reconstruct())            # PQ is more accurate at the same bits


def test_pq_lowers_output_error_vs_int3():
    W = _correlated((1024, 512), rank=8, seed=7)
    rng = np.random.default_rng(8)
    x = rng.standard_normal((32, 512)).astype(np.float32)
    int3 = QuantizedWeightStore(W, bits=3, huffman=True, device=_DEV)
    pq = ProductQuantizedWeightStore(W, subdim=4, codebook_bits=6, device=_DEV)
    oerr = lambda R: float(np.mean((x @ R.T - x @ W.T) ** 2))
    assert oerr(pq.reconstruct()) < oerr(int3.reconstruct())         # lower output error at matched rate


def test_usable_below_one_and_a_half_bits():
    # the capability scalar quant lacks: a usable operating point in the sub-1.5-bit regime
    W = _correlated((1024, 512), rank=8, seed=9)
    pq = ProductQuantizedWeightStore(W, subdim=8, codebook_bits=8, device=_DEV)
    assert pq.bits_per_weight() < 1.5
    # far better reconstruction than scalar int2 (which is the only scalar option near this rate)
    int2 = QuantizedWeightStore(W, bits=2, huffman=True, device=_DEV)
    assert float(np.mean((pq.reconstruct() - W) ** 2)) < float(np.mean((int2.reconstruct() - W) ** 2))


def test_serialise_round_trip():
    W = _correlated((256, 128), seed=10)
    st = ProductQuantizedWeightStore(W, subdim=4, codebook_bits=8, device=_DEV)
    st2 = ProductQuantizedWeightStore.load(st.save(), device=_DEV)
    assert np.array_equal(st.reconstruct(), st2.reconstruct())
    idx = np.random.default_rng(11).integers(0, st.n, 300)
    assert np.allclose(st2.fetch(idx), st.reconstruct().ravel()[idx], atol=1e-6)


def test_size_must_be_multiple_of_subdim():
    W = np.zeros((10, 7), np.float32)                            # 70 not divisible by 4
    try:
        ProductQuantizedWeightStore(W, subdim=4, codebook_bits=6, device=_DEV)
        assert False, "expected ValueError"
    except ValueError:
        pass
