"""Fold-and-merge — the mechanism where folding *is* the compression.

The idea in one picture: take a strand of symbols and **wrap it onto a cylinder** whose
circumference is the strand's natural period ``p`` — so coil *r* is the slice ``[r·p : (r+1)·p]``
and every coil lands directly on the one below it. Now look down each column: wherever a cell
equals the cell in the coil beneath it, the two **merge** — we keep it once and record a single
"same" bit; only where they differ do we store the actual value. A periodic / self-similar strand
therefore collapses to *one* template coil plus a sea of "same" bits and a few differences.

Then we do it again: the template coil is itself wrapped and merged, and again, layer by layer —
the strand condensing into a **chromosome** of nested coils. Unwrapping replays the differences
outward from the innermost core, exactly. That is the whole codec:

* **lossless** — ``tolerance = 0``; ``unfold(fold(x)) == x``.
* **lossy** — ``tolerance = q``; cells within ``q`` of the one below are merged and their small
  difference discarded, so more of the strand collapses, at bounded error.

``fold_levels`` also returns the per-layer merge pattern, which is exactly what the animation
replays to *show* the wrapping happen in time.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .varint import pack_uvarints, read_uvarint, unpack_uvarints, write_uvarint

_MAGIC = b"WFOLD1"
_FLAG_LOSSY = 0x01


@dataclass
class Level:
    """One coil layer: wrap the strand at period ``p`` and merge each cell onto the one below."""
    period: int
    length: int                 # length of the strand at this layer (before folding)
    same: np.ndarray            # bool[length - period]: did cell k merge with cell k-period?
    diffs: np.ndarray           # int[]: the values of the cells that did NOT merge

    @property
    def merged(self) -> int:
        return int(self.same.sum())


def best_period(sym: np.ndarray, tol: int, min_p: int = 2, max_p: Optional[int] = None,
                threshold: float = 0.34) -> Tuple[Optional[int], float]:
    """Find the fold period: the shift ``p`` whose coil-on-coil agreement is highest.

    Returns ``(period, agreement)`` or ``(None, best)`` if nothing folds well enough to help."""
    n = len(sym)
    if max_p is None:
        max_p = n // 2
    best_p, best_score = None, 0.0
    for p in range(min_p, max_p + 1):
        a = sym[p:]
        b = sym[:-p]
        score = float(np.mean(np.abs(a.astype(np.int64) - b.astype(np.int64)) <= tol))
        if score > best_score + 1e-9:
            best_score, best_p = score, p
    if best_p is None or best_score < threshold:
        return None, best_score
    return best_p, best_score


def wrap_merge(sym: np.ndarray, p: int, tol: int) -> Level:
    """Wrap ``sym`` at period ``p`` and merge each cell onto the cell one coil below."""
    n = len(sym)
    a = sym[p:].astype(np.int64)
    b = sym[:-p].astype(np.int64)
    same = np.abs(a - b) <= tol
    diffs = sym[p:][~same].astype(np.int32)
    return Level(period=p, length=n, same=same, diffs=diffs)


def wrap_unmerge(base: np.ndarray, level: Level) -> np.ndarray:
    """Invert :func:`wrap_merge`: rebuild the strand from the template coil + merge pattern."""
    n = level.length
    p = level.period
    out = np.empty(n, np.int32)
    out[:p] = base
    di = 0
    same = level.same
    diffs = level.diffs
    for k in range(p, n):
        if same[k - p]:
            out[k] = out[k - p]
        else:
            out[k] = diffs[di]
            di += 1
    return out


def fold_levels(sym, tol: int = 0, min_period: int = 2, min_gain: float = 0.02):
    """Recursively wrap-and-merge into nested coils.

    Returns ``(core, levels)`` — the innermost template coil plus the list of coil layers, outer
    first. Stops when a further fold would not shrink the serialized form."""
    cur = np.asarray(sym, np.int32)
    levels: List[Level] = []
    while len(cur) > 2 * min_period:
        p, score = best_period(cur, tol, min_p=min_period)
        if p is None:
            break
        lvl = wrap_merge(cur, p, tol)
        # gain: bits saved by merging vs. the cost of the new base coil + diffs. Fold only if it
        # actually helps (a merged cell costs ~1 bit instead of a whole symbol).
        merged = lvl.merged
        saved = merged * 8 - (len(cur) - p)          # bytes→bits handled in codec; rough proxy
        if merged == 0 or saved <= min_gain * len(cur) * 8:
            break
        levels.append(lvl)
        cur = cur[:p].copy()                          # the template coil becomes the next strand
    return cur, levels


def unfold_levels(core: np.ndarray, levels: List[Level]) -> np.ndarray:
    """Invert :func:`fold_levels` — unwrap from the innermost core outward."""
    cur = np.asarray(core, np.int32)
    for lvl in reversed(levels):
        cur = wrap_unmerge(cur, lvl)
    return cur


# --------------------------------------------------------------------------- serialization
def _pack_bits(bits: np.ndarray) -> bytes:
    return np.packbits(bits.astype(np.uint8)).tobytes()


def _unpack_bits(buf: bytes, n: int) -> np.ndarray:
    return np.unpackbits(np.frombuffer(buf, np.uint8), count=n).astype(bool)


def compress(data: bytes, mode: str = "lossless", tol: int = 6) -> bytes:
    """Fold ``data`` into its nested chromosome and serialize it to a self-describing blob.

    ``mode="lossy"`` merges coils that agree within ``tol`` (bounded error), collapsing more of
    the strand; ``mode="lossless"`` (``tol=0``) reconstructs exactly."""
    lossy = mode == "lossy"
    t = tol if lossy else 0
    sym = np.frombuffer(data, np.uint8).astype(np.int32)
    core, levels = fold_levels(sym, tol=t)

    out = bytearray()
    out += _MAGIC
    out.append(_FLAG_LOSSY if lossy else 0)
    write_uvarint(out, len(data))
    write_uvarint(out, len(levels))
    for lvl in levels:                                   # outer → inner
        write_uvarint(out, lvl.period)
        write_uvarint(out, lvl.length)
        write_uvarint(out, len(lvl.diffs))
        out += pack_uvarints([int(x) for x in lvl.diffs])
        bits = _pack_bits(lvl.same)
        write_uvarint(out, len(bits))
        out += bits
    write_uvarint(out, len(core))
    out += pack_uvarints([int(x) for x in core])
    return bytes(out)


def decompress(blob: bytes) -> bytes:
    """Invert :func:`compress` — unwrap the chromosome back to bytes."""
    if blob[:len(_MAGIC)] != _MAGIC:
        raise ValueError("not a WFOLD1 blob")
    pos = len(_MAGIC)
    _flags = blob[pos]; pos += 1
    n, pos = read_uvarint(blob, pos)
    nlev, pos = read_uvarint(blob, pos)
    levels: List[Level] = []
    for _ in range(nlev):
        p, pos = read_uvarint(blob, pos)
        length, pos = read_uvarint(blob, pos)
        ndiff, pos = read_uvarint(blob, pos)
        diffs, pos = unpack_uvarints(blob, pos, ndiff)
        nbytes, pos = read_uvarint(blob, pos)
        same = _unpack_bits(blob[pos:pos + nbytes], length - p); pos += nbytes
        levels.append(Level(period=p, length=length,
                            same=same, diffs=np.array(diffs, np.int32)))
    clen, pos = read_uvarint(blob, pos)
    core, pos = unpack_uvarints(blob, pos, clen)
    sym = unfold_levels(np.array(core, np.int32), levels)
    return bytes(int(x) & 0xFF for x in sym[:n])
