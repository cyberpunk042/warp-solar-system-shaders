"""Folding — reshape a linear sequence into a lattice and read it back along a space-filling curve.

A fold is a **reversible permutation** of the sequence. It compresses nothing on its own; it
*rearranges* so that structure the coiler can exploit becomes adjacent. "Fold the card into a
cube": lay the stream row-major into a square (2-D) or cube (3-D), then read the cells out along a
Morton (Z-order) curve so that neighbours in the fold — not just neighbours in the original line —
end up next to each other. The best fold for a given input is found by trying each and keeping
whichever coils smallest (see ``codec``); every fold is exactly invertible, so this never costs
correctness.

Folds are identified by a small integer tag so the header can record which one was used.
"""

import math
from typing import List

FOLD_NONE = 0
FOLD_MORTON2D = 1
FOLD_MORTON3D = 2

FOLD_NAMES = {FOLD_NONE: "none", FOLD_MORTON2D: "morton2d", FOLD_MORTON3D: "morton3d"}


def _part1by1(n: int) -> int:
    """Spread the low bits of ``n`` into even positions (2-D Morton interleave helper)."""
    n &= 0xFFFFFFFF
    n = (n | (n << 16)) & 0x0000FFFF0000FFFF
    n = (n | (n << 8)) & 0x00FF00FF00FF00FF
    n = (n | (n << 4)) & 0x0F0F0F0F0F0F0F0F
    n = (n | (n << 2)) & 0x3333333333333333
    n = (n | (n << 1)) & 0x5555555555555555
    return n


def _morton2(x: int, y: int) -> int:
    return _part1by1(x) | (_part1by1(y) << 1)


def _morton3(x: int, y: int, z: int) -> int:
    def spread(v: int) -> int:
        r = 0
        for i in range(21):
            r |= (v & (1 << i)) << (2 * i)
        return r

    return spread(x) | (spread(y) << 1) | (spread(z) << 2)


def _side_2d(n: int) -> int:
    s = 1
    while s * s < n:
        s <<= 1                                   # power-of-two side keeps Morton codes dense
    return s


def _side_3d(n: int) -> int:
    s = 1
    while s * s * s < n:
        s <<= 1
    return s


def fold(seq: List[int], kind: int) -> List[int]:
    """Return the sequence read out of the lattice along the fold's curve (a permutation).

    Length is preserved: cells that fall outside the original length are dropped on the way out
    and restored as zero-padding on the way back, so ``unfold(fold(seq, k), k, len(seq))`` is
    ``seq`` for every ``kind``."""
    n = len(seq)
    if kind == FOLD_NONE or n <= 2:
        return list(seq)

    if kind == FOLD_MORTON2D:
        s = _side_2d(n)
        order = []
        for m in range(s * s):
            x = _demort2(m, 0)
            y = _demort2(m, 1)
            idx = y * s + x
            if idx < n:
                order.append(idx)
        return [seq[i] for i in order]

    if kind == FOLD_MORTON3D:
        s = _side_3d(n)
        order = []
        for m in range(s * s * s):
            x, y, z = _demort3(m)
            idx = (z * s + y) * s + x
            if idx < n:
                order.append(idx)
        return [seq[i] for i in order]

    raise ValueError(f"unknown fold kind {kind}")


def unfold(folded: List[int], kind: int, n: int) -> List[int]:
    """Invert :func:`fold` back to the original linear order of length ``n``."""
    if kind == FOLD_NONE or n <= 2:
        return list(folded)

    out = [0] * n
    if kind == FOLD_MORTON2D:
        s = _side_2d(n)
        k = 0
        for m in range(s * s):
            x = _demort2(m, 0)
            y = _demort2(m, 1)
            idx = y * s + x
            if idx < n:
                out[idx] = folded[k]
                k += 1
        return out

    if kind == FOLD_MORTON3D:
        s = _side_3d(n)
        k = 0
        for m in range(s * s * s):
            x, y, z = _demort3(m)
            idx = (z * s + y) * s + x
            if idx < n:
                out[idx] = folded[k]
                k += 1
        return out

    raise ValueError(f"unknown fold kind {kind}")


def _demort2(m: int, axis: int) -> int:
    """Extract axis ``0`` (x) or ``1`` (y) from a 2-D Morton code."""
    v = 0
    bit = 0
    m >>= axis
    while m:
        v |= (m & 1) << bit
        m >>= 2
        bit += 1
    return v


def _demort3(m: int):
    x = y = z = 0
    bit = 0
    while m:
        x |= (m & 1) << bit
        y |= ((m >> 1) & 1) << bit
        z |= ((m >> 2) & 1) << bit
        m >>= 3
        bit += 1
    return x, y, z
