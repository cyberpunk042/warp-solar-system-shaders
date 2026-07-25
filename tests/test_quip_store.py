"""QuIP (incoherence × error-feedback): row-addressable, and the composed lever beats either parent alone."""
import numpy as np

import warp as wp

from warp_compress.gptq_store import GPTQWeightStore
from warp_compress.hadamard_store import HadamardQuantStore
from warp_compress.quip_store import QuIPWeightStore
from warp_compress.weight_store import QuantizedWeightStore

_DEV = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"


def _outlier_weights_and_calib(out=512, inn=256, n=4096, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((inn, 16))
    X = (rng.standard_normal((n, 16)) @ A.T + 0.3 * rng.standard_normal((n, inn))).astype(np.float32)
    W = ((rng.standard_normal((out, 16)) @ rng.standard_normal((16, inn))) / np.sqrt(16 * inn)
         + rng.standard_normal((out, inn)) / np.sqrt(inn)).astype(np.float32)
    m = rng.random(W.shape) < 0.003
    W[m] += rng.standard_normal(int(m.sum())).astype(np.float32) * 3.0        # outlier-heavy
    return W, X


def test_rows_are_addressable():
    W, X = _outlier_weights_and_calib(seed=1)
    st = QuIPWeightStore(W, X, bits=4, device=_DEV)
    R = st.reconstruct()
    idx = np.random.default_rng(2).integers(0, W.shape[0], 16)
    assert np.allclose(st.fetch_rows(idx), R[idx], atol=1e-6)                # per-row decode == full decode


def test_reconstruct_is_lossless_over_rotated_codes():
    W, X = _outlier_weights_and_calib(128, 64, 1024, seed=3)
    st = QuIPWeightStore(W, X, bits=4, device=_DEV)
    from warp_compress.hadamard_store import _rht
    manual = _rht(st._rotated_recon(), st.signs, st.block, inverse=True)
    assert np.allclose(st.reconstruct(), manual, atol=1e-6)                  # exactly rotate-back(dequant codes)


def test_stacks_super_additively_on_outliers():
    W, X = _outlier_weights_and_calib(seed=4)
    oerr = lambda Q: float(np.mean((X @ Q.T - X @ W.T) ** 2))
    rtn = QuantizedWeightStore(W, bits=4, huffman=True, device=_DEV).reconstruct().reshape(W.shape)
    had = HadamardQuantStore(W, bits=4, device=_DEV).reconstruct()
    gptq = GPTQWeightStore(W, X, bits=4, device=_DEV).reconstruct()
    quip = QuIPWeightStore(W, X, bits=4, device=_DEV).reconstruct()
    e_rtn, e_had, e_gptq, e_quip = oerr(rtn), oerr(had), oerr(gptq), oerr(quip)
    assert e_quip < e_had and e_quip < e_gptq                                # beats EITHER parent alone
    assert e_quip < 0.25 * e_rtn                                             # and a big win over plain RTN


def test_gptq_alone_barely_helps_on_outliers_but_quip_does():
    # the finding: outliers blow the scale so GPTQ-alone ~ RTN; rotating first unlocks it
    W, X = _outlier_weights_and_calib(seed=5)
    oerr = lambda Q: float(np.mean((X @ Q.T - X @ W.T) ** 2))
    rtn = QuantizedWeightStore(W, bits=4, huffman=True, device=_DEV).reconstruct().reshape(W.shape)
    gptq = GPTQWeightStore(W, X, bits=4, device=_DEV).reconstruct()
    quip = QuIPWeightStore(W, X, bits=4, device=_DEV).reconstruct()
    assert oerr(gptq) > 0.5 * oerr(rtn)                                      # GPTQ alone ~ RTN here (outliers)
    assert oerr(quip) < 0.3 * oerr(gptq)                                     # QuIP unlocks the error-feedback


def test_rate_is_int_class_and_seed_deterministic():
    W, X = _outlier_weights_and_calib(seed=6)
    a = QuIPWeightStore(W, X, bits=4, seed=9, device=_DEV)
    b = QuIPWeightStore(W, X, bits=4, seed=9, device=_DEV)
    assert np.array_equal(a.signs, b.signs)                                  # rotation determined by seed
    assert np.allclose(a.reconstruct(), b.reconstruct(), atol=1e-6)
    assert a.bits_per_weight() < 8.0                                         # int4-class rate + a free seed


def test_shape_validation():
    W = np.zeros((8, 6), np.float32)
    for badX in (np.zeros((10, 5), np.float32), np.zeros(6, np.float32)):
        try:
            QuIPWeightStore(W, badX, bits=4, device=_DEV)
            assert False, "expected ValueError"
        except ValueError:
            pass
