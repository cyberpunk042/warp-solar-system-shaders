"""awq — activation-aware weight scaling (AWQ, arXiv 2306.00978) that ChromoFold sits on top of.

v1/the outlier lever quantize weights as they are; AWQ makes low-bit *usable* by first scaling the **salient
input channels** (the ones multiplied by large activations) UP before quantization, so they keep more resolution
— then folding the inverse scale into the activations, leaving the output unchanged. It's a per-input-channel
diagonal, so unlike an incoherence *rotation* it **preserves ChromoFold's random access** (fetch one weight,
divide by its channel scale). ChromoFold then entropy-codes the AWQ-scaled quantized weights, losslessly.

    awq_scale(W, act_scale, bits, group_size)  -> the per-input-channel scale (grid-searched over α)
    QuantizedWeightStore(W, channel_scale=s)   -> quantize W·diag(s), undo diag(s) at dequant (lossless, RA)

`act_scale[j]` = mean |x_j| over a calibration batch (one forward pass). α=0 (no scaling = plain quant) is in
the grid, so AWQ never loses on the proxy. Measured on real gpt2 in `bench_awq.py`. Run: python -m warp_compress.awq
"""
from __future__ import annotations

import numpy as np


def _fake_quant(W, bits: int, group_size):
    """Quantize→dequantize `W` exactly as QuantizedWeightStore would (per-tensor or per-group symmetric int)."""
    lim = (1 << (bits - 1)) - 1
    flat = np.asarray(W, np.float32).ravel()
    if group_size is None:
        scale = float(np.abs(flat).max()) / lim + 1e-12
        deq = np.clip(np.round(flat / scale), -lim, lim) * scale
    else:
        g = int(group_size); n = flat.shape[0]; ng = (n + g - 1) // g
        pad = np.zeros(ng * g, np.float32); pad[:n] = flat
        sc = np.abs(pad.reshape(ng, g)).max(1) / lim + 1e-12
        deq = (np.clip(np.round(pad / np.repeat(sc, g)), -lim, lim) * np.repeat(sc, g))[:n]
    return deq.reshape(np.asarray(W).shape)


def awq_scale(W, act_scale, bits: int = 4, group_size: "int | None" = None, grid: int = 20):
    """Per-input-channel scale `s` (len = in_features) for W (out×in) that minimises the activation-weighted
    quantization error ‖(W − Q(W·diag(s))·diag(1/s))·diag(act)‖². Returns (s, best_alpha, best_err)."""
    W = np.asarray(W, np.float32)
    act = np.clip(np.asarray(act_scale, np.float32), 1e-6, None)          # mean |x| per input channel
    best_s, best_alpha, best_err = np.ones_like(act), 0.0, np.inf
    for alpha in np.linspace(0.0, 1.0, grid):
        s = act ** alpha
        s = s / np.sqrt(s.max() * s.min() + 1e-12)                        # keep the geometric scale ~1
        Wh = _fake_quant(W * s[None, :], bits, group_size) / s[None, :]   # quantize scaled, undo the scale
        err = float(np.mean(((W - Wh) * act[None, :]) ** 2))
        if err < best_err:
            best_s, best_alpha, best_err = s.astype(np.float32), float(alpha), err
    return best_s, best_alpha, best_err


def _demo():
    # synthetic layer with a few high-activation "salient" input channels (the AWQ regime)
    rng = np.random.default_rng(0)
    out_f, in_f = 512, 512
    W = (rng.standard_normal((out_f, in_f)) / np.sqrt(in_f)).astype(np.float32)
    act = np.abs(rng.standard_normal(in_f)).astype(np.float32)
    act[rng.choice(in_f, 8, replace=False)] *= 30.0                       # salient channels
    s, alpha, err_awq = awq_scale(W, act, bits=4, grid=21)

    def out_err(scale):
        Wh = _fake_quant(W * scale[None, :], 4, None) / scale[None, :]
        x = act[None, :] * rng.standard_normal((64, in_f)).astype(np.float32)
        return float(np.mean((x @ (W - Wh).T) ** 2))

    plain = out_err(np.ones(in_f, np.float32))
    awq = out_err(s)
    print(f"AWQ scale search: best α={alpha:.2f}\n")
    print(f"  int4 output MSE  plain {plain:.3e}   AWQ {awq:.3e}   => {plain/awq:.2f}× lower")
    print("=> scale salient input channels up before quantizing (activations take the inverse), so the channels "
          "that matter keep resolution. A per-channel diagonal -> ChromoFold keeps random access + entropy-codes\n"
          "   the result. Helps when a FEW channels are salient (this synthetic layer); on real gpt2 group-wise "
          "scaling dominates and AWQ adds little — an honest negative, measured in bench_awq.py.")


if __name__ == "__main__":
    _demo()
