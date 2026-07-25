"""Low-rank weight codec (+ compressed residual): addressable factors, lossless over the residual codec."""
import numpy as np

import warp as wp

from warp_compress.lowrank_store import LowRankWeightStore
from warp_compress.weight_store import QuantizedWeightStore

_DEV = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"


def _lowrank_plus_noise(shape=(1024, 384), rank=16, seed=0):
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((shape[0], rank)).astype(np.float32)
    V = rng.standard_normal((rank, shape[1])).astype(np.float32)
    return ((U @ V) / np.sqrt(rank * shape[1]) + rng.standard_normal(shape) / np.sqrt(shape[1])).astype(np.float32)


def test_fetch_rows_matches_reconstruct():
    W = _lowrank_plus_noise(seed=1)
    for res in (None, "int", "pq"):
        st = LowRankWeightStore(W, rank=16, residual=res, device=_DEV)
        R = st.reconstruct()
        rows = np.array([0, 3, 11, 1023])
        assert np.allclose(st.fetch_rows(rows), R[rows], atol=1e-4)   # addressable == full decode (lossless/codec)


def test_pure_lowrank_captures_structure_cheaply():
    W = _lowrank_plus_noise(rank=16, seed=2)
    st = LowRankWeightStore(W, rank=16, residual=None, device=_DEV)
    assert st.bits_per_weight() < 1.5                                 # thin factors -> very few bits
    # the reconstruction should recover most of the SIGNAL energy (structure), even dropping the noise
    signal = st.reconstruct()
    resid_energy = float(np.mean((W - signal) ** 2))
    assert resid_energy < float(np.mean(W ** 2))                      # captured a real chunk of the tensor


def test_composed_is_competitive_with_int4_at_matched_bits():
    # honest: at matched bits low-rank + PQ residual is COMPETITIVE with int4 (roughly tied, usually a touch
    # better on output error) — not a blanket win. Assert within-20% on both, which holds across seeds.
    W = _lowrank_plus_noise((2048, 512), rank=16, seed=3)
    rng = np.random.default_rng(4)
    x = rng.standard_normal((48, 512)).astype(np.float32)
    int4 = QuantizedWeightStore(W, bits=4, huffman=True, device=_DEV)
    lr = LowRankWeightStore(W, rank=16, residual="pq", device=_DEV)
    assert abs(lr.bits_per_weight() - int4.bits_per_weight()) < 0.5   # comparable rate
    mse = lambda R: float(np.mean((R - W) ** 2))
    oerr = lambda R: float(np.mean((x @ R.T - x @ W.T) ** 2))
    assert mse(lr.reconstruct()) < 1.2 * mse(int4.reconstruct())     # competitive on MSE
    assert oerr(lr.reconstruct()) < 1.2 * oerr(int4.reconstruct())   # competitive on output error


def test_residual_lowers_error_vs_pure_lowrank():
    W = _lowrank_plus_noise(seed=5)
    pure = LowRankWeightStore(W, rank=16, residual=None, device=_DEV)
    withres = LowRankWeightStore(W, rank=16, residual="int", device=_DEV)
    assert np.mean((withres.reconstruct() - W) ** 2) < np.mean((pure.reconstruct() - W) ** 2)


def test_rank_is_capped_and_factor_shapes():
    W = _lowrank_plus_noise((100, 60), seed=6)
    st = LowRankWeightStore(W, rank=999, residual=None, device=_DEV)   # rank capped to min(out, in)
    assert st.rank == 60
    assert st.A.shape == (100, 60) and st.B.shape == (60, 60)


def test_higher_rank_reconstructs_better():
    W = _lowrank_plus_noise((512, 256), rank=32, seed=7)
    lo = LowRankWeightStore(W, rank=8, residual=None, device=_DEV)
    hi = LowRankWeightStore(W, rank=32, residual=None, device=_DEV)
    assert np.mean((hi.reconstruct() - W) ** 2) < np.mean((lo.reconstruct() - W) ** 2)


def test_invalid_residual_raises():
    W = _lowrank_plus_noise((64, 64), seed=8)
    try:
        LowRankWeightStore(W, rank=8, residual="bogus", device=_DEV)
        assert False, "expected ValueError"
    except ValueError:
        pass
