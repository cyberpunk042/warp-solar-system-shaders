"""GPTQ error-feedback quant: addressable, lossless over its codes, measured output-error win vs round-to-nearest."""
import numpy as np

import warp as wp

from warp_compress.gptq_store import GPTQWeightStore
from warp_compress.weight_store import QuantizedWeightStore

_DEV = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"


def _weights_and_calib(out=512, inn=256, n=4096, rank=16, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((inn, rank))
    X = (rng.standard_normal((n, rank)) @ A.T + 0.3 * rng.standard_normal((n, inn))).astype(np.float32)  # correlated
    W = ((rng.standard_normal((out, rank)) @ rng.standard_normal((rank, inn))) / np.sqrt(rank * inn)
         + rng.standard_normal((out, inn)) / np.sqrt(inn)).astype(np.float32)
    return W, X


def test_fetch_matches_reconstruct():
    W, X = _weights_and_calib(seed=1)
    st = GPTQWeightStore(W, X, bits=4, device=_DEV)
    R = st.reconstruct().ravel()
    idx = np.random.default_rng(2).integers(0, st.n, 500)
    assert np.allclose(st.fetch(idx), R[idx], atol=1e-6)             # addressable == full decode


def test_beats_round_to_nearest_on_output_error():
    W, X = _weights_and_calib(seed=3)
    oerr = lambda Q: float(np.mean((X @ Q.T - X @ W.T) ** 2))
    rtn = QuantizedWeightStore(W, bits=4, huffman=True, device=_DEV).reconstruct().reshape(W.shape)
    gptq = GPTQWeightStore(W, X, bits=4, device=_DEV)
    assert oerr(gptq.reconstruct()) < 0.6 * oerr(rtn)               # error-feedback: substantially lower output err


def test_same_rate_as_rtn_int():
    W, X = _weights_and_calib(seed=4)
    rtn = QuantizedWeightStore(W, bits=4, huffman=True, device=_DEV)
    gptq = GPTQWeightStore(W, X, bits=4, device=_DEV)
    assert abs(gptq.bits_per_weight() - rtn.bits_per_weight()) < 0.5  # same int-class rate (win is quality, not bits)
    assert gptq.bits_per_weight() < 8.0


def test_reconstruct_is_lossless_over_its_codes():
    # the store holds integer levels; decoding them back must be exact (lossless over the chosen quantization)
    W, X = _weights_and_calib(128, 64, 1024, seed=5)
    st = GPTQWeightStore(W, X, bits=4, device=_DEV)
    levels = st.wm.access(np.arange(st.n, dtype=np.int64))
    manual = ((levels.astype(np.float32) - st._lim) * st.scale).reshape(W.shape)
    assert np.array_equal(st.reconstruct(), manual)                 # decode is exactly (level-lim)*scale


def test_levels_within_range():
    W, X = _weights_and_calib(seed=6)
    st = GPTQWeightStore(W, X, bits=4, device=_DEV)
    levels = st.wm.access(np.arange(st.n, dtype=np.int64))
    assert levels.min() >= 0 and levels.max() <= 2 * st._lim        # valid int4 levels [0, 2*lim]


def test_shape_validation():
    W = np.zeros((8, 6), np.float32)
    for badX in (np.zeros((10, 5), np.float32), np.zeros(6, np.float32)):   # wrong in-dim; wrong ndim
        try:
            GPTQWeightStore(W, badX, bits=4, device=_DEV)
            assert False, "expected ValueError"
        except ValueError:
            pass


def test_save_load_roundtrip():
    W, X = _weights_and_calib(128, 64, 1024, seed=7)
    st = GPTQWeightStore(W, X, bits=4, device=_DEV)
    st2 = GPTQWeightStore.load(st.save(), device=_DEV)
    assert np.allclose(st.reconstruct(), st2.reconstruct(), atol=1e-6)   # reload reproduces the same weights
