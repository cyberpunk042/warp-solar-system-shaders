"""The codec — orchestrates Fold → Coil → Pack (and back), lossless or lossy.

``compress`` turns bytes into a self-describing blob; ``decompress`` inverts it exactly (lossless)
or to the quantized approximation (lossy). The fold is chosen by *search*: every candidate fold is
coiled and the one whose chromosome packs smallest wins — folding can only ever reorder, never
corrupt, so the search is free of risk.

Blob layout (all integers LEB128 unless noted)::

    magic  "WARP1"          5 bytes
    flags   1 byte          bit0: lossy
    fold    1 byte          fold kind tag (see fold.py)
    base    uvarint         literal-alphabet size
    n       uvarint         number of symbols before folding
    q       uvarint         lossy quantisation step (0 when lossless)
    R       uvarint         number of rules (nucleosomes)
    2R      uvarints        rule pairs a0 b0 a1 b1 ...   (rule id = base + index)
    T       uvarint         top-strand length
    T       uvarints        top symbols
"""

import zlib
from typing import Dict

from . import fold as _fold
from .chromosome import coil, uncoil, Chromosome
from .varint import pack_uvarints, read_uvarint, unpack_uvarints, write_uvarint

_MAGIC = b"WARP1"
_FLAG_LOSSY = 0x01

_FOLD_CANDIDATES = (_fold.FOLD_NONE, _fold.FOLD_MORTON2D, _fold.FOLD_MORTON3D)


def _quantize(data: bytes, q: int):
    """Map bytes onto ``q``-wide levels; return ``(levels, base)`` with a shrunk alphabet."""
    base = (255 // q) + 1
    levels = [b // q for b in data]
    return levels, base


def _dequantize(levels, q: int) -> bytes:
    half = q // 2
    return bytes(min(lvl * q + half, 255) for lvl in levels)


def _serialize(chrom: Chromosome, fold_kind: int, n: int, lossy: bool, q: int) -> bytes:
    out = bytearray()
    out += _MAGIC
    out.append(_FLAG_LOSSY if lossy else 0)
    out.append(fold_kind)
    write_uvarint(out, chrom.base)
    write_uvarint(out, n)
    write_uvarint(out, q if lossy else 0)
    write_uvarint(out, len(chrom.rules))
    flat = []
    for a, b in chrom.rules:
        flat.append(a)
        flat.append(b)
    out += pack_uvarints(flat)
    write_uvarint(out, len(chrom.top))
    out += pack_uvarints(chrom.top)
    return bytes(out)


def _coil_symbols(symbols, base, fold_kind):
    folded = _fold.fold(symbols, fold_kind)
    return coil(folded, base=base)


def compress(data: bytes, mode: str = "lossless", q: int = 8, fold_kind="auto") -> bytes:
    """Compress ``data``. ``mode`` is ``"lossless"`` or ``"lossy"`` (with quantisation step ``q``).

    ``fold_kind="auto"`` searches the fold candidates and keeps the smallest; pass an explicit
    tag from :mod:`warp_compress.fold` to force one."""
    lossy = mode == "lossy"
    if lossy:
        symbols, base = _quantize(data, max(1, q))
    else:
        symbols, base, q = list(data), 256, 0
    n = len(symbols)

    if fold_kind == "auto":
        candidates = _FOLD_CANDIDATES
    else:
        candidates = (int(fold_kind),)

    best_blob = None
    for kind in candidates:
        chrom = _coil_symbols(symbols, base, kind)
        blob = _serialize(chrom, kind, n, lossy, q)
        if best_blob is None or len(blob) < len(best_blob):
            best_blob = blob
    return best_blob


def decompress(blob: bytes) -> bytes:
    """Invert :func:`compress` — exact for lossless blobs, quantized approximation for lossy."""
    if blob[:5] != _MAGIC:
        raise ValueError("not a WARP1 blob")
    pos = 5
    flags = blob[pos]; pos += 1
    fold_kind = blob[pos]; pos += 1
    base, pos = read_uvarint(blob, pos)
    n, pos = read_uvarint(blob, pos)
    q, pos = read_uvarint(blob, pos)
    r, pos = read_uvarint(blob, pos)
    flat, pos = unpack_uvarints(blob, pos, 2 * r)
    rules = [(flat[2 * i], flat[2 * i + 1]) for i in range(r)]
    t, pos = read_uvarint(blob, pos)
    top, pos = unpack_uvarints(blob, pos, t)

    folded = uncoil(Chromosome(base=base, rules=rules, top=top))
    symbols = _fold.unfold(folded, fold_kind, n)

    if flags & _FLAG_LOSSY:
        return _dequantize(symbols, q)
    return bytes(symbols)


def describe(blob: bytes) -> Dict[str, object]:
    """Parse a blob's header and report its shape + size (for diagnostics and the demo)."""
    pos = 5
    flags = blob[pos]; pos += 1
    fold_kind = blob[pos]; pos += 1
    base, pos = read_uvarint(blob, pos)
    n, pos = read_uvarint(blob, pos)
    q, pos = read_uvarint(blob, pos)
    r, pos = read_uvarint(blob, pos)
    _, pos = unpack_uvarints(blob, pos, 2 * r)
    t, pos = read_uvarint(blob, pos)
    return {
        "lossy": bool(flags & _FLAG_LOSSY),
        "fold": _fold.FOLD_NAMES.get(fold_kind, str(fold_kind)),
        "base": base,
        "symbols": n,
        "quant_step": q,
        "nucleosomes": r,
        "top_symbols": t,
        "packed_bytes": len(blob),
        "packed_plus_zlib": len(zlib.compress(blob, 9)),
    }
