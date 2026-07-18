"""C1 — merge → cube: dedup identical pieces of the card, keep a location index (lossless).

Operator spec (2026-07-14, verbatim): *"one compression for example can just merge the same thing
together and have digit to represent the locations of the vsrious same element that also grow a but the
cube part."*

Merge the same thing together, and keep digits for where each same element goes. Concretely, on the
**real card**: sample it to a 3-D occupancy grid, cut it into small **blocks**, and **merge every
identical block into one stored copy** (a dictionary of the unique pieces). A grid of **digits** — the
location index — records, for each block position, which unique piece belongs there. Because the board
is full of repeats (identical GDDR7 packages, identical VRM chokes, the regular PCB), a handful of
unique pieces plus the index reconstruct the whole card **exactly** (lossless). The index grid is "the
cube part": a compact cube of digits that **grows** with the number of distinct pieces while the merged
dictionary stays small.

Reconstruction places `dictionary[index[p]]` at every block position — exact. Verified in
``tests/test_mergecube.py``: `decompress(compress(x)) == x`, and the measured size ratio.
"""

import numpy as np

from .foldcube import sample_card


def _pad_to(occ, b):
    nx, ny, nz = occ.shape
    px, py, pz = (-nx) % b, (-ny) % b, (-nz) % b
    if px or py or pz:
        occ = np.pad(occ, ((0, px), (0, py), (0, pz)))
    return occ


def compress(occ, block=4):
    """Dedup identical `block`-sized cubes; return (dictionary, index grid, meta) — lossless."""
    orig_shape = occ.shape
    occ = _pad_to(occ, block)
    nx, ny, nz = occ.shape
    nbx, nby, nbz = nx // block, ny // block, nz // block
    blocks = (occ.reshape(nbx, block, nby, block, nbz, block)
                 .transpose(0, 2, 4, 1, 3, 5)
                 .reshape(nbx * nby * nbz, block ** 3))
    unique, inverse = np.unique(blocks, axis=0, return_inverse=True)   # merge identical blocks
    index = inverse.reshape(nbx, nby, nbz).astype(np.int32)            # the location-index "cube"
    return unique, index, {"orig_shape": orig_shape, "padded": (nx, ny, nz), "block": block}


def decompress(unique, index, meta):
    """Place dictionary[index] at every block position — exact reconstruction."""
    b = meta["block"]
    nbx, nby, nbz = index.shape
    blocks = unique[index.reshape(-1)]                                  # (n_blocks, b^3)
    occ = (blocks.reshape(nbx, nby, nbz, b, b, b)
                 .transpose(0, 3, 1, 4, 2, 5)
                 .reshape(nbx * b, nby * b, nbz * b))
    ox, oy, oz = meta["orig_shape"]
    return occ[:ox, :oy, :oz]


def ratio(occ, unique, index, meta):
    """Lossless size ratio: original bits / (dictionary bits + location-index digit bits)."""
    b = meta["block"]
    orig_bits = int(np.prod(meta["padded"]))                           # 1 bit / voxel
    n_unique = max(len(unique), 2)
    id_bits = int(np.ceil(np.log2(n_unique)))                          # digits to name each unique piece
    dict_bits = unique.size                                             # the merged cores, stored once
    index_bits = index.size * id_bits                                  # the growing location-index cube
    return orig_bits / (dict_bits + index_bits)


def compress_card(block=4):
    """Convenience: sample the real board, merge-compress, verify lossless, return a report."""
    occ = sample_card()
    unique, index, meta = compress(occ, block=block)
    back = decompress(unique, index, meta)
    return {
        "lossless": bool(np.array_equal(back, occ)),
        "n_blocks": int(index.size),
        "n_unique": int(len(unique)),
        "index_shape": index.shape,
        "ratio": ratio(occ, unique, index, meta),
        "block": block,
    }
