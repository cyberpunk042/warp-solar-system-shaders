"""C5 — super-elastic fold: stack more matrix-op layers to squeeze the residual further.

Operator spec (2026-07-25, verbatim): *"the super-elastic mode that skeeze even more in the folding,
more layer of matrix operation to solve the puzzle to resolve our own representation of the data."*

Where C4 (:mod:`warp_compress.grouped_delta`) folds a group once — reference + a single rank-``r``
factored residual "atom" — **super-elastic folds the leftover again, and again**. Each layer takes the
*residual the previous layers could not represent*, factors it (rank-``r``), int8-quantizes the factors,
and subtracts its reconstruction; the next layer works on the smaller leftover. ``layers`` is the
**elastic depth dial**: more layers => more matrix-op work at decode ("solve the puzzle") => a tighter
squeeze of the residual. Decode = reference + the sum of every layer's reconstruction.

This is residual (a.k.a. multi-stage) quantization applied to the grouped-delta atom: each layer gets a
*fresh per-matrix scale* adapted to its shrinking residual, so L small-rank layers can spend a byte
budget more efficiently than one big-rank layer — the hypothesis this module MEASURES (P7 — report the
result either way; a negative here means "one big layer was as good", and that is a finding).

Honesty (ChromoFold P4): every layer is a *named lossy* step; ``final_residual=True`` additionally keeps
the exact leftover so the whole stack round-trips **bit-exact** (super-elastic then becomes a lossless
re-encoding whose lossy layers only decide how much of the residual is entropy-coded raw vs factored).

Pure numpy + zlib. Verified in ``tests/test_super_elastic.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .grouped_delta import (
    _entropy_bytes,
    _grid,
    _minify_int,
    _quant_i8,
    build_reference,
)


@dataclass
class SuperElastic:
    """A reference + a stack of int8 low-rank residual layers (+ optional exact final residual)."""

    shape: tuple
    dtype: np.dtype
    mode: str
    n: int
    reference: np.ndarray
    layers: list                    # list of (uq, uq_scale, vq, vq_scale) int8 factor pairs
    final_residual: np.ndarray = None  # exact int leftover (n, d) for the lossless product; else None


def _fold_layer(resid, rank):
    """One layer: rank-`rank` SVD of the residual, folded into int8 factors. Returns (uq,us,vq,vs)."""
    u, s, vt = np.linalg.svd(resid, full_matrices=False)
    r = int(min(rank, u.shape[1]))
    uq, us = _quant_i8(u[:, :r] * s[:r])
    vq, vs = _quant_i8(vt[:r])
    return uq, us, vq, vs


def _layer_recon(layer):
    uq, us, vq, vs = layer
    return (uq.astype(np.float64) * us) @ (vq.astype(np.float64) * vs)


def compress(group, layers=3, rank=4, mode="centroid", final_residual=False):
    """Fold `group` with `layers` stacked int8 rank-`rank` residual layers (the elastic dial)."""
    g = np.asarray(group)
    if g.ndim < 2:
        raise ValueError("group must be (n, *tensor_shape)")
    n, shape = g.shape[0], g.shape[1:]
    ref = build_reference(g, mode)
    resid = (g.astype(np.float64) - ref.astype(np.float64)).reshape(n, -1)

    stack = []
    for _ in range(layers):
        layer = _fold_layer(resid, rank)
        stack.append(layer)
        resid = resid - _layer_recon(layer)      # pass the shrinking leftover to the next layer

    fr = None
    if final_residual:
        fr = np.rint(resid).astype(np.int64)     # exact remaining integer residual -> lossless
    return SuperElastic(shape, g.dtype, mode, n, ref, stack, fr)


def decompress(se):
    """Reconstruct = reference + sum of every layer's reconstruction (+ exact leftover if kept)."""
    lo, hi = _grid(se.dtype)
    recon = se.reference.astype(np.float64).reshape(-1) + sum(_layer_recon(l) for l in se.layers)
    recon = np.rint(recon).astype(np.int64)
    if se.final_residual is not None:
        recon = recon + se.final_residual
    recon = np.clip(recon, lo, hi).astype(se.dtype)
    return recon.reshape((se.n, *se.shape))


def encoded_bytes(se):
    """Entropy-coded size: reference + every layer's int8 factors (+ exact leftover if kept)."""
    total = _entropy_bytes(se.reference)
    for uq, us, vq, vs in se.layers:
        total += _entropy_bytes(uq) + _entropy_bytes(vq) + 8   # +2 float32 scales
    if se.final_residual is not None:
        total += _entropy_bytes(_minify_int(se.final_residual))
    return total


def mean_abs_err(group, se):
    g = np.asarray(group).astype(np.int64)
    return float(np.abs(g - decompress(se).astype(np.int64)).mean())


def baseline_bytes(group):
    """Independent storage: each tensor entropy-coded on its own (no folding)."""
    g = np.asarray(group)
    return sum(_entropy_bytes(g[i]) for i in range(g.shape[0]))


def compress_group(group, layers=3, rank=4, mode="centroid", final_residual=False):
    """Convenience report dict (mirrors grouped_delta.compress_group)."""
    g = np.asarray(group)
    se = compress(g, layers=layers, rank=rank, mode=mode, final_residual=final_residual)
    back = decompress(se)
    return {
        "layers": layers,
        "rank": rank,
        "lossless": bool(final_residual and np.array_equal(back, g)),
        "atom_bytes": encoded_bytes(se),
        "baseline_bytes": baseline_bytes(g),
        "ratio_vs_baseline": baseline_bytes(g) / max(encoded_bytes(se), 1),
        "mean_abs_err": mean_abs_err(g, se),
    }
