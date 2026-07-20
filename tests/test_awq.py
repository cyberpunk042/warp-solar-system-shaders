"""AWQ activation-aware scale search: it never loses to plain quant, and helps when channels are salient."""
import numpy as np

from warp_compress.awq import _fake_quant, awq_scale


def _layer(out_f=128, in_f=128, salient=4, seed=0):
    rng = np.random.default_rng(seed)
    W = (rng.standard_normal((out_f, in_f)) / np.sqrt(in_f)).astype(np.float32)
    act = np.abs(rng.standard_normal(in_f)).astype(np.float32)
    act[rng.choice(in_f, salient, replace=False)] *= 30.0
    return W, act


def test_scale_length_and_positivity():
    W, act = _layer(seed=1)
    s, alpha, err = awq_scale(W, act, bits=4)
    assert s.shape == (W.shape[1],) and np.all(s > 0) and 0.0 <= alpha <= 1.0


def test_awq_never_worse_than_plain_on_the_proxy():
    W, act = _layer(seed=2)
    s, _, err_awq = awq_scale(W, act, bits=4, grid=21)
    Wh_plain = _fake_quant(W, 4, None)
    err_plain = float(np.mean(((W - Wh_plain) * act[None, :]) ** 2))
    assert err_awq <= err_plain + 1e-12                 # alpha=0 (no scaling) is in the grid


def test_awq_lowers_output_error_with_salient_channels():
    W, act = _layer(salient=6, seed=3)
    s = awq_scale(W, act, bits=4)[0]
    rng = np.random.default_rng(4)
    x = act[None, :] * rng.standard_normal((64, W.shape[1])).astype(np.float32)
    e_awq = np.mean((x @ (W - _fake_quant(W * s[None, :], 4, None) / s[None, :]).T) ** 2)
    e_plain = np.mean((x @ (W - _fake_quant(W, 4, None)).T) ** 2)
    assert e_awq < e_plain


def test_uniform_activations_give_near_identity_scale():
    W, _ = _layer(seed=5)
    s = awq_scale(W, np.ones(W.shape[1], np.float32), bits=4)[0]
    assert np.allclose(s, 1.0, atol=1e-5)               # no salient channels -> nothing to protect
