"""bench_levers torch-free core: apply_lever reconstructs every lever + raises LeverIncompatible on bad shapes."""
import numpy as np

import warp as wp

from warp_compress.bench_levers import LEVERS, LeverIncompatible, apply_lever, _selftest

_DEV = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"


def _weights(out=256, inn=128, rank=8, seed=0):
    rng = np.random.default_rng(seed)
    return ((rng.standard_normal((out, rank)) @ rng.standard_normal((rank, inn))) / np.sqrt(rank * inn)
            + rng.standard_normal((out, inn)) / np.sqrt(inn)).astype(np.float32)   # low-rank + noise, model-like


def test_every_lever_reconstructs_shape_and_positive_rate():
    W = _weights(seed=1)
    denom = float((W ** 2).mean())
    for name in LEVERS:
        recon, bpw = apply_lever(W, name, device=_DEV)
        assert recon.shape == W.shape and recon.dtype == np.float32
        assert np.isfinite(recon).all()
        assert 0.0 < bpw < 40.0                                  # a real measured rate, not nominal width
        assert ((recon - W) ** 2).mean() / denom < 1.0           # lossy but tracks the weights (not garbage)


def test_int_rates_ordered_and_below_fp16():
    W = _weights(seed=2)
    b8 = apply_lever(W, "int8", device=_DEV)[1]
    b4 = apply_lever(W, "int4", device=_DEV)[1]
    assert b4 < b8 < 16.0                                        # int4 cheaper than int8, both below fp16


def test_outliers_costs_bits_but_lowers_error():
    W = _weights(seed=3)
    r4, b4 = apply_lever(W, "int4", device=_DEV)
    ro, bo = apply_lever(W, "int4_outliers", device=_DEV)
    err = lambda R: float(((R - W) ** 2).mean())
    assert bo > b4                                               # keeping fp16 outliers spends more bits
    assert err(ro) < err(r4)                                     # ...and buys lower reconstruction error


def test_pq_family_beats_int4_rate():
    W = _weights(512, 256, seed=4)                              # one shared codebook amortizes over the whole tensor
    b_int4 = apply_lever(W, "int4", device=_DEV)[1]
    for name in ("pq", "rvq"):
        assert apply_lever(W, name, device=_DEV)[1] < b_int4     # sub-int4 bits/weight (single-codebook levers)


def test_subspace_pq_rate_needs_a_wide_matrix():
    # honest "it depends": subspace-PQ fits ONE codebook per column block, so its rate only amortizes when there
    # are many rows to spread those codebooks over. Narrow -> the per-subspace codebooks are a tax; wide -> a win.
    b_narrow = apply_lever(_weights(256, 256, seed=8), "subspace_pq", device=_DEV)[1]
    b_wide = apply_lever(_weights(4096, 256, seed=8), "subspace_pq", device=_DEV)[1]
    assert b_wide < b_narrow                                     # more rows amortize the codebooks -> fewer bits/wt


def test_hadamard_lever_selectable_and_helps_on_outliers():
    rng = np.random.default_rng(20)
    W = (rng.standard_normal((1024, 256)) / 16).astype(np.float32)
    m = rng.random(W.shape) < 0.003
    W[m] += rng.standard_normal(int(m.sum())).astype(np.float32) * 3.0        # outlier-heavy
    r_int4, _ = apply_lever(W, "int4", device=_DEV)
    r_had, b_had = apply_lever(W, "hadamard", device=_DEV)
    assert r_had.shape == W.shape and b_had < 8.0
    assert ((r_had - W) ** 2).mean() < ((r_int4 - W) ** 2).mean()             # incoherence lowers int4 error


def test_incompatible_shape_raises_not_crashes():
    Wbad = np.random.default_rng(5).standard_normal((7, 5)).astype(np.float32)   # size 35 odd, in=5
    for name in ("pq", "rvq", "subspace_pq"):
        try:
            apply_lever(Wbad, name, device=_DEV)
            assert False, f"{name} should have raised LeverIncompatible"
        except LeverIncompatible:
            pass
    # int levers take any shape
    for name in ("int8", "int4", "int4_outliers"):
        recon, _ = apply_lever(Wbad, name, device=_DEV)
        assert recon.shape == Wbad.shape


def test_unknown_lever_rejected():
    try:
        apply_lever(_weights(seed=6), "nonesuch")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_overrides_change_the_rate():
    W = _weights(512, 256, seed=7)
    b_default = apply_lever(W, "pq", device=_DEV)[1]             # subdim=4
    b_wide = apply_lever(W, "pq", subdim=8, device=_DEV)[1]     # 2x sub-vector -> half the codes -> fewer bits
    assert b_wide < b_default


def test_selftest_passes():
    assert _selftest() == 0                                      # the module's own torch-free self-test is green
