"""LEB128 variable-length integers — the byte packing the chromosome is serialized with.

Small symbol ids cost one byte, large ones grow gracefully; this is what makes the coiled
grammar's size (and therefore the compression ratio) a real, measurable number rather than a
count of abstract symbols.
"""

from typing import List, Tuple


def write_uvarint(out: bytearray, value: int) -> None:
    """Append ``value`` (>= 0) to ``out`` as an unsigned LEB128 varint."""
    if value < 0:
        raise ValueError("uvarint is unsigned")
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return


def read_uvarint(buf: bytes, pos: int) -> Tuple[int, int]:
    """Read an unsigned LEB128 varint from ``buf`` at ``pos``; return ``(value, new_pos)``."""
    result = 0
    shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return result, pos
        shift += 7


def pack_uvarints(values: List[int]) -> bytes:
    out = bytearray()
    for v in values:
        write_uvarint(out, v)
    return bytes(out)


def unpack_uvarints(buf: bytes, pos: int, count: int) -> Tuple[List[int], int]:
    out: List[int] = []
    for _ in range(count):
        v, pos = read_uvarint(buf, pos)
        out.append(v)
    return out, pos
