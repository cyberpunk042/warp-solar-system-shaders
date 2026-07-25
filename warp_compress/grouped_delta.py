"""C4 — grouped delta superposition: fold similar tensors into a shared reference + a factored residual atom.

Operator spec (2026-07-25, verbatim): *"the idea is that if you can have similar things and group them
and rather them being defined by a delta from the root or from a third state of it you then having your
delta, then you can overlap them into a single atom like shape and play with the matrix to compact them
into this new model of fit the most inside GPU memory. trying to find the sweet spot for how much to rely
on Processor VS Memory and tried to regain other performances by doing so."* + *"An even smarter
transformation, we keep and keep on ravaging with our matrix operations"*

Group N similar tensors (quantized weight tiles, LoRA adapters, MoE experts — same shape). Pick a
**reference** the group is expressed against: a designated **root** member, or a computed **"third state"**
centroid that is *no single member*. Each member becomes reference + residual, ``T_i = R + D_i``. Stack the
residuals and **overlap them into one "atom"** with matrix operations — either the exact residual set
(lossless), or a rank-``r`` factorization ``D ~= U @ diag(S) @ Vt`` (the lossy "matrix compaction"; the
rank ``r`` is the processor-vs-memory dial).

Two products, kept honest (this is the CPU oracle for ChromoFold, whose P4 is *lossless over the chosen
quantization*):

  * **lossless** (``rank=None``) — reference (grid-snapped) + EXACT integer residuals, entropy-coded.
    ``decompress`` is bit-exact. The win comes ONLY from correlation: similar members => near-zero
    residuals => low entropy. No correlation => no win.
  * **approx** (``rank=r``) — rank-``r`` residual reconstruction; a NAMED lossy layer with a measured
    error budget, never hidden. As ``r`` grows, error falls and size rises — the P1 dial made explicit.

The decisive question this prototype answers (ChromoFold P7/P10 — measured, not asserted; report
negatives): does grouping actually beat storing each tensor independently? For correlated groups, yes;
for uncorrelated groups it does NOT, and ``ratio_vs_baseline <= 1`` says so honestly. Verified in
``tests/test_grouped_delta.py`` (run: ``python -m tests.test_grouped_delta``).
"""

import zlib
from dataclasses import dataclass

import numpy as np


@dataclass
class GroupedDelta:
    """The compressed 'atom' for a group of similar tensors.

    ``reference`` is the shared root/centroid (grid-snapped, one copy for the whole group). For the
    lossless product ``residuals`` holds the exact integer ``T_i - R`` (shape ``(n, *shape)``) and the
    factor fields are ``None``. For the approx product the residual set is replaced by its rank-``r``
    factorization ``u``/``s``/``vt`` and ``residuals`` is ``None``.
    """

    shape: tuple           # per-member tensor shape
    dtype: np.dtype        # the quantized grid the members live on
    mode: str              # "root" | "centroid"
    n: int                 # group size
    reference: np.ndarray  # R, shape == shape, dtype == dtype
    residuals: np.ndarray  # exact int residuals (n, prod(shape)) for lossless; else None
    rank: int              # r for approx; None for lossless
    u: np.ndarray          # (n, r) approx factor; None for lossless
    s: np.ndarray          # (r,)  approx singular values; None for lossless
    vt: np.ndarray         # (r, d) approx factor; None for lossless


def _grid(dtype):
    """The legal value range of the quantization grid the tensors live on."""
    info = np.iinfo(dtype)
    return info.min, info.max


def build_reference(group, mode="centroid"):
    """Compute the shared reference the group is expressed against.

    ``root`` = the first member (a designated existing tensor, like M8's base). ``centroid`` = the
    grid-snapped mean — the operator's *"third state"*, a synthesized tensor that is no single member.
    """
    g = np.asarray(group)
    if mode == "root":
        return g[0].astype(g.dtype, copy=True)
    if mode == "centroid":
        lo, hi = _grid(g.dtype)
        return np.clip(np.rint(g.mean(axis=0)), lo, hi).astype(g.dtype)
    raise ValueError(f"unknown reference mode {mode!r} (use 'root' or 'centroid')")


def compress(group, mode="centroid", rank=None):
    """Fold a group of same-shape tensors into a :class:`GroupedDelta`.

    ``rank=None`` => the lossless product (exact residuals). ``rank=r`` => the approx product (rank-``r``
    residual factorization — a named lossy layer).
    """
    g = np.asarray(group)
    if g.ndim < 2:
        raise ValueError("group must be (n, *tensor_shape) with n >= 1 tensors")
    n = g.shape[0]
    shape = g.shape[1:]
    ref = build_reference(g, mode)
    # residuals in a signed width that can hold the full grid span (uint8 diffs -> int16, etc.)
    resid = (g.astype(np.int64) - ref.astype(np.int64)).reshape(n, -1)

    if rank is None:
        return GroupedDelta(shape, g.dtype, mode, n, ref, resid.astype(np.int64),
                            None, None, None, None)

    r = int(min(rank, n, resid.shape[1]))
    u, s, vt = np.linalg.svd(resid.astype(np.float64), full_matrices=False)
    return GroupedDelta(shape, g.dtype, mode, n, ref, None,
                        r, u[:, :r].copy(), s[:r].copy(), vt[:r].copy())


def decompress(gd):
    """Reconstruct the group. Bit-exact for the lossless product; grid-snapped approx otherwise."""
    lo, hi = _grid(gd.dtype)
    ref = gd.reference.astype(np.int64).reshape(-1)
    if gd.residuals is not None:
        recon = ref + gd.residuals            # exact
    else:
        approx = gd.u @ np.diag(gd.s) @ gd.vt  # rank-r residual
        recon = ref + np.rint(approx).astype(np.int64)
    recon = np.clip(recon, lo, hi).astype(gd.dtype)
    return recon.reshape((gd.n, *gd.shape))


# ---- honest bit-accounting: measure with a real entropy coder (zlib), as codec.describe does ----

def _entropy_bytes(arr):
    """Real entropy-coded size of an integer array (zlib -9), the repo's comparison coder."""
    return len(zlib.compress(np.ascontiguousarray(arr).tobytes(), 9))


def _minify_int(arr):
    """Downcast an integer array to the smallest signed dtype that holds it losslessly.

    Residuals are computed in int64 for headroom, but stored/entropy-coded at their true width — a
    fair comparison against a baseline whose members are their native (narrow) dtype, and how a real
    codec would emit them (an int8 residual is 1 byte, not 8).
    """
    lo, hi = int(arr.min()), int(arr.max()) if arr.size else (0, 0)
    for dt in (np.int8, np.int16, np.int32):
        info = np.iinfo(dt)
        if lo >= info.min and hi <= info.max:
            return arr.astype(dt)
    return arr


def encoded_bytes(gd):
    """Entropy-coded size of the atom: reference once + (exact residuals | rank-r factors)."""
    total = _entropy_bytes(gd.reference)
    if gd.residuals is not None:
        total += _entropy_bytes(_minify_int(gd.residuals))
    else:
        # float32 factors are the honest storage for the approx atom
        total += _entropy_bytes(gd.u.astype(np.float32))
        total += _entropy_bytes(gd.s.astype(np.float32))
        total += _entropy_bytes(gd.vt.astype(np.float32))
    return total


def baseline_bytes(group):
    """Independent storage: each tensor entropy-coded on its own (no grouping)."""
    g = np.asarray(group)
    return sum(_entropy_bytes(g[i]) for i in range(g.shape[0]))


def ratio(group, gd):
    """Raw bytes / atom bytes — compression vs the uncompressed group."""
    raw = np.asarray(group).size * np.asarray(group).dtype.itemsize
    return raw / max(encoded_bytes(gd), 1)


def ratio_vs_baseline(group, gd):
    """Atom bytes vs independent-storage bytes. > 1 => grouping WON; <= 1 => it did not (report it)."""
    return baseline_bytes(group) / max(encoded_bytes(gd), 1)


def mean_abs_err(group, gd):
    """Mean absolute reconstruction error (0.0 for the lossless product)."""
    g = np.asarray(group).astype(np.int64)
    back = decompress(gd).astype(np.int64)
    return float(np.abs(g - back).mean())


def compress_group(group, mode="centroid", rank=None):
    """Convenience: fold, verify round-trip, return a report dict (mirrors ``mergecube.compress_card``)."""
    g = np.asarray(group)
    gd = compress(g, mode=mode, rank=rank)
    back = decompress(gd)
    return {
        "lossless": bool(rank is None and np.array_equal(back, g)),
        "mode": mode,
        "n": int(g.shape[0]),
        "shape": tuple(int(x) for x in g.shape[1:]),
        "rank": gd.rank,
        "ratio": ratio(g, gd),
        "ratio_vs_baseline": ratio_vs_baseline(g, gd),
        "mean_abs_err": mean_abs_err(g, gd),
        "atom_bytes": encoded_bytes(gd),
        "baseline_bytes": baseline_bytes(g),
    }
