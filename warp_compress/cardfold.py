"""Card-fold — fold a flat card into a cube, the 2-D sibling of the chromosome wrap.

Where ``wrapfold`` wraps a 1-D strand into a chromosome, this folds a 2-D **card** like paper:
fold it in half (one half lands on the other), and wherever the two stacked cells match, they
**merge** — kept once, recorded as a single "same" bit; only the cells that differ store a value.
Fold again on the other axis, and again, alternating — the card halving each time and thickening
into layers, condensing toward a compact **cube** (the fundamental tile the card is built from).
Unfolding replays the differences outward, exactly.

* **lossless** — ``tolerance = 0``; ``unfold_card(fold_card(x)) == x``.
* **lossy** — ``tolerance = q``; cells within ``q`` merge and the small difference is dropped.

Folds are **mirror** folds (the far half flips onto the near half, as real paper does), so a card
with mirror symmetry collapses to its fundamental quadrant; ``fold_levels_card`` also returns the
per-fold merge mask, which the ``warp_card`` scene replays to *show* the card fold in time.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .varint import pack_uvarints, read_uvarint, unpack_uvarints, write_uvarint

_MAGIC = b"WCARD1"
_FLAG_LOSSY = 0x01


@dataclass
class CardLevel:
    """One fold: axis (1 = width, 0 = height), the pre-fold shape, and the merge pattern."""
    axis: int
    h: int
    w: int
    same: np.ndarray            # bool[half] flattened row-major over the folded-away half
    diffs: np.ndarray           # values of the cells that did not merge


def _agree(grid: np.ndarray, axis: int, tol: int) -> float:
    """Fraction of cells that match their mirror partner across the centre crease of ``axis``."""
    if axis == 1:
        w = grid.shape[1]
        half = w // 2
        if half == 0:
            return 0.0
        a = grid[:, :half]
        b = grid[:, w - 1:w - 1 - half:-1]              # mirror of the right half
    else:
        h = grid.shape[0]
        half = h // 2
        if half == 0:
            return 0.0
        a = grid[:half, :]
        b = grid[h - 1:h - 1 - half:-1, :]
    return float(np.mean(np.abs(a.astype(np.int64) - b.astype(np.int64)) <= tol))


def fold_once(grid: np.ndarray, axis: int, tol: int) -> Tuple[np.ndarray, CardLevel]:
    """Mirror-fold ``grid`` in half along ``axis`` and merge matching cells. Returns (kept, level)."""
    h, w = grid.shape
    if axis == 1:
        half = w // 2
        kept = grid[:, :half].copy()
        mirror = grid[:, w - 1:w - 1 - half:-1]         # what folds onto kept
    else:
        half = h // 2
        kept = grid[:half, :].copy()
        mirror = grid[h - 1:h - 1 - half:-1, :]
    same = (np.abs(kept.astype(np.int64) - mirror.astype(np.int64)) <= tol)
    diffs = mirror[~same].astype(np.int32).ravel()
    return kept, CardLevel(axis=axis, h=h, w=w, same=same.ravel(), diffs=diffs)


def unfold_once(kept: np.ndarray, lvl: CardLevel) -> np.ndarray:
    """Invert :func:`fold_once` — rebuild the full card from the kept half + merge pattern."""
    h, w = lvl.h, lvl.w
    out = np.empty((h, w), np.int32)
    same = lvl.same.reshape(kept.shape)
    mirror = np.where(same, kept, 0).astype(np.int32)
    if lvl.diffs.size:
        mirror[~same] = lvl.diffs
    if lvl.axis == 1:
        half = w // 2
        out[:, :half] = kept
        out[:, w - 1:w - 1 - half:-1] = mirror
        if w % 2:                                        # odd centre column is part of kept side
            out[:, half] = kept[:, -1] if half < kept.shape[1] else kept[:, half - 1]
    else:
        half = h // 2
        out[:half, :] = kept
        out[h - 1:h - 1 - half:-1, :] = mirror
        if h % 2:
            out[half, :] = kept[-1, :] if half < kept.shape[0] else kept[half - 1, :]
    return out


def fold_levels_card(grid, tol: int = 0, min_side: int = 2, threshold: float = 0.55):
    """Recursively fold the card in half, alternating axes, while folds keep merging well.

    Returns ``(core, levels)`` — the small residual tile plus the folds, outermost first."""
    cur = np.asarray(grid, np.int32)
    levels: List[CardLevel] = []
    axis = 1
    while cur.shape[0] > min_side or cur.shape[1] > min_side:
        h, w = cur.shape
        # pick the axis (of the two) that folds best and is still large enough
        cand = [a for a in (1, 0) if (cur.shape[1 - a] if False else (w if a == 1 else h)) >= 2 * min_side]
        cand = [a for a in (axis, 1 - axis) if (w if a == 1 else h) >= 2]
        best_a, best_s = None, 0.0
        for a in cand:
            s = _agree(cur, a, tol)
            if s > best_s:
                best_s, best_a = s, a
        if best_a is None or best_s < threshold:
            break
        cur, lvl = fold_once(cur, best_a, tol)
        levels.append(lvl)
        axis = 1 - best_a
    return cur, levels


def unfold_levels_card(core: np.ndarray, levels: List[CardLevel]) -> np.ndarray:
    cur = np.asarray(core, np.int32)
    for lvl in reversed(levels):
        cur = unfold_once(cur, lvl)
    return cur


# --------------------------------------------------------------------------- serialization
def compress(grid: np.ndarray, mode: str = "lossless", tol: int = 6) -> bytes:
    """Fold a 2-D card into its cube and serialize. ``grid`` is an ``HxW`` array of 0..255."""
    lossy = mode == "lossy"
    t = tol if lossy else 0
    grid = np.asarray(grid, np.int32)
    h, w = grid.shape
    core, levels = fold_levels_card(grid, tol=t)
    out = bytearray()
    out += _MAGIC
    out.append(_FLAG_LOSSY if lossy else 0)
    write_uvarint(out, h)
    write_uvarint(out, w)
    write_uvarint(out, len(levels))
    for lvl in levels:
        out.append(lvl.axis)
        write_uvarint(out, lvl.h)
        write_uvarint(out, lvl.w)
        write_uvarint(out, len(lvl.diffs))
        out += pack_uvarints([int(x) for x in lvl.diffs])
        bits = np.packbits(lvl.same.astype(np.uint8)).tobytes()
        write_uvarint(out, len(bits))
        out += bits
        write_uvarint(out, lvl.same.size)
    ch, cw = core.shape
    write_uvarint(out, ch)
    write_uvarint(out, cw)
    out += pack_uvarints([int(x) for x in core.ravel()])
    return bytes(out)


def decompress(blob: bytes) -> np.ndarray:
    """Invert :func:`compress` — unfold the cube back into the full card (``HxW`` uint8 array)."""
    if blob[:len(_MAGIC)] != _MAGIC:
        raise ValueError("not a WCARD1 blob")
    pos = len(_MAGIC)
    _flags = blob[pos]; pos += 1
    _h, pos = read_uvarint(blob, pos)
    _w, pos = read_uvarint(blob, pos)
    nlev, pos = read_uvarint(blob, pos)
    levels: List[CardLevel] = []
    for _ in range(nlev):
        axis = blob[pos]; pos += 1
        lh, pos = read_uvarint(blob, pos)
        lw, pos = read_uvarint(blob, pos)
        ndiff, pos = read_uvarint(blob, pos)
        diffs, pos = unpack_uvarints(blob, pos, ndiff)
        nbytes, pos = read_uvarint(blob, pos)
        raw = blob[pos:pos + nbytes]; pos += nbytes
        nsame, pos = read_uvarint(blob, pos)
        same = np.unpackbits(np.frombuffer(raw, np.uint8), count=nsame).astype(bool)
        levels.append(CardLevel(axis=axis, h=lh, w=lw, same=same,
                                diffs=np.array(diffs, np.int32)))
    ch, pos = read_uvarint(blob, pos)
    cw, pos = read_uvarint(blob, pos)
    core, pos = unpack_uvarints(blob, pos, ch * cw)
    grid = unfold_levels_card(np.array(core, np.int32).reshape(ch, cw), levels)
    return (grid & 0xFF).astype(np.uint8)
