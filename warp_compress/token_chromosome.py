"""token_chromosome — the genome compression engine as callable math (no rendering).

Extracts the mechanism proven by ``warp_shaders/genome/thread.py``: a token sequence is stored as a
**chromosome** = (a content book of the V unique tokens) + (an id stream) + (a closed-form, order- and
locality-preserving embedding Phi that maps rank -> a point in a compact box). Because Phi is procedural
and invertible, you get **token-by-token navigation inside the compressed form**:

    at(r)          O(1) position of token r in the compact box            (forward Phi)
    invert(x)      O(1) rank at a box position                            (inverse Phi)
    next(r)/prev   a LOCAL hop (neighbours in rank stay neighbours in box)
    token(r)       the token content at rank r                           (book[ids[r]])
    decompress(a,b) the token run in [a,b)

Phi here is a d-dimensional **Hilbert curve** (Skilling's algorithm) — the canonical locality-preserving
space-filling curve, with exact O(bits*d) index<->point maps in both directions. This replaces the
boustrophedon read order of the DNA engine with its provably-invertible form. The DNA folds
(base pair -> helix -> ... -> chromatid) are a *hierarchical* variant of the same idea; see
``docs/genome_compression_math.md``.

Design goal: geometry as an O(1) index into a clustered, addressable, compressed sequence.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

# --------------------------------------------------------------------------------------------------
# Hilbert curve (Skilling 2004): integer index <-> integer coordinates in [0, 2**bits)**dim.
# Locality-preserving: consecutive indices are always adjacent (differ by 1 in exactly one axis).
# --------------------------------------------------------------------------------------------------

def _index_to_transpose(index: int, bits: int, dim: int) -> list:
    """De-interleave a scalar Hilbert index into the `dim`-coordinate 'transpose' form (Skilling)."""
    x = [0] * dim
    for i in range(bits):
        for j in range(dim):
            bit = (index >> (bits * dim - 1 - (i * dim + j))) & 1
            x[j] = (x[j] << 1) | bit
    return x


def _transpose_to_index(x: list, bits: int, dim: int) -> int:
    """Interleave the transpose form back into a scalar Hilbert index."""
    h = 0
    for i in range(bits):
        for j in range(dim):
            h = (h << 1) | ((x[j] >> (bits - 1 - i)) & 1)
    return h


def _transpose_to_axes(x: list, bits: int, dim: int) -> None:
    """In-place: transpose (Hilbert) form -> spatial axes."""
    n = 1 << bits
    t = x[dim - 1] >> 1
    for i in range(dim - 1, 0, -1):
        x[i] ^= x[i - 1]
    x[0] ^= t
    q = 2
    while q != n:
        p = q - 1
        for i in range(dim - 1, -1, -1):
            if x[i] & q:
                x[0] ^= p
            else:
                t = (x[0] ^ x[i]) & p
                x[0] ^= t
                x[i] ^= t
        q <<= 1


def _axes_to_transpose(x: list, bits: int, dim: int) -> None:
    """In-place: spatial axes -> transpose (Hilbert) form (inverse of _transpose_to_axes)."""
    m = 1 << (bits - 1)
    q = m
    while q > 1:
        p = q - 1
        for i in range(dim):
            if x[i] & q:
                x[0] ^= p
            else:
                t = (x[0] ^ x[i]) & p
                x[0] ^= t
                x[i] ^= t
        q >>= 1
    for i in range(1, dim):
        x[i] ^= x[i - 1]
    t = 0
    q = m
    while q > 1:
        if x[dim - 1] & q:
            t ^= q - 1
        q >>= 1
    for i in range(dim):
        x[i] ^= t


def hilbert_point(index: int, bits: int, dim: int) -> list:
    """Hilbert index -> integer coordinates in [0, 2**bits)**dim. O(bits*dim)."""
    x = _index_to_transpose(int(index), bits, dim)
    _transpose_to_axes(x, bits, dim)
    return x


def hilbert_index(coords, bits: int, dim: int) -> int:
    """Integer coordinates -> Hilbert index (exact inverse of hilbert_point). O(bits*dim)."""
    x = [int(c) for c in coords]
    _axes_to_transpose(x, bits, dim)
    return _transpose_to_index(x, bits, dim)


# --------------------------------------------------------------------------------------------------
# The chromosome: merge-code (content dedup) + Hilbert embedding.
# --------------------------------------------------------------------------------------------------

@dataclasses.dataclass
class Chromosome:
    """A token sequence compressed to (book, id stream, Hilbert embedding). All navigation is closed-form.

    ``book`` (V, …) the unique token contents; ``ids`` (N,) the per-rank index into the book; ``bits``/``dim``
    the Hilbert curve so that 2**(bits*dim) >= N. Positions are never stored — always evaluated."""

    book: np.ndarray
    ids: np.ndarray
    bits: int
    dim: int

    @property
    def n(self) -> int:
        return int(self.ids.shape[0])

    @property
    def side(self) -> int:
        return (1 << self.bits) - 1

    # --- navigation inside the compressed form ---
    def at(self, r: int):
        """O(1) position of token r as a point in the unit box [0,1]**dim (the compact chromosome)."""
        c = hilbert_point(int(r), self.bits, self.dim)
        return np.asarray(c, np.float64) / max(self.side, 1)

    def invert(self, x) -> int:
        """O(1) rank at a box position (nearest lattice site). Exact when x = at(r)."""
        c = np.clip(np.round(np.asarray(x, np.float64) * self.side), 0, self.side).astype(np.int64)
        r = hilbert_index([int(v) for v in c], self.bits, self.dim)
        return int(min(max(r, 0), self.n - 1))

    def next(self, r: int) -> int:
        return (int(r) + 1) % self.n

    def prev(self, r: int) -> int:
        return (int(r) - 1) % self.n

    def token(self, r: int):
        return self.book[self.ids[int(r)]]

    def decompress(self, lo: int = 0, hi: int | None = None) -> np.ndarray:
        hi = self.n if hi is None else hi
        return self.book[self.ids[lo:hi]]

    # --- accounting ---
    def rate_bits(self) -> dict:
        """Stored bits: content book + run-length-encoded id stream + the (tiny) map constants. Positions
        and order cost nothing — they are procedural."""
        v = int(self.book.shape[0])
        content_bits = self.book.size * self.book.dtype.itemsize * 8
        runs = 1 + int(np.count_nonzero(np.diff(self.ids)))          # RLE run count
        id_bits = runs * (max(1, math.ceil(math.log2(max(v, 2)))) + 16)   # (symbol + run length)
        return dict(V=v, N=self.n, content_bits=content_bits, id_bits=id_bits,
                    map_bits=64, total_bits=content_bits + id_bits + 64)


def compress(tokens, dim: int = 3) -> Chromosome:
    """Compress an ordered token sequence into a :class:`Chromosome`. ``tokens`` is a 1-D array of hashable
    symbols (ints). Content is deduped (merge codec); order is embedded on a Hilbert curve of `dim` dims."""
    tokens = np.asarray(tokens)
    book, ids = np.unique(tokens, return_inverse=True)
    ids = ids.astype(np.int64).ravel()
    n = int(ids.shape[0])
    bits = max(1, math.ceil(math.log2(max(n, 2)) / dim))            # 2**(bits*dim) >= n
    return Chromosome(book=book, ids=ids, bits=bits, dim=dim)


# --------------------------------------------------------------------------------------------------
# Self-test / demo: round-trip, exact invertibility, locality, and a rate readout.
# --------------------------------------------------------------------------------------------------

def _selftest():
    # a token stream with heavy repetition (like the merge-coded card): V unique, N total
    rng = np.random.default_rng(7)
    blocks = rng.integers(0, 40, size=600)                          # 40 distinct "types"
    seq = np.repeat(blocks, rng.integers(3, 12, size=blocks.shape))  # runs of repeats
    ch = compress(seq, dim=3)
    N = ch.n

    # 1) lossless: decompress == original
    assert np.array_equal(ch.decompress(), seq), "round-trip failed"

    # 2) exact O(1) invertibility: invert(at(r)) == r for every r
    bad = sum(1 for r in range(N) if ch.invert(ch.at(r)) != r)
    assert bad == 0, f"{bad} ranks did not invert"

    # 3) locality: consecutive ranks are adjacent in the box (Hilbert -> exactly 1/side apart)
    pts = np.stack([ch.at(r) for r in range(N)])
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    max_step = float(steps.max())

    rate = ch.rate_bits()
    raw_bits = N * (max(1, math.ceil(math.log2(rate["V"]))) )       # N symbols, log2(V) bits each
    print(f"N={N}  V={rate['V']}  dim={ch.dim}  bits/axis={ch.bits}  box_side={ch.side + 1}")
    print(f"round-trip: OK   invert(at(r))==r for all {N} ranks: OK")
    print(f"locality: max neighbour step = {max_step:.4f}  (Hilbert ideal = {1.0 / ch.side:.4f})")
    print(f"rate: content={rate['content_bits']}b  ids(RLE)={rate['id_bits']}b  "
          f"total={rate['total_bits']}b   vs raw id stream ~{raw_bits}b   "
          f"=> {raw_bits / max(rate['total_bits'], 1):.2f}x")
    print("navigation demo:  at(0)=", np.round(ch.at(0), 3),
          " next->", ch.next(0), " token(0)=", ch.token(0),
          " invert(at(1234))=", ch.invert(ch.at(min(1234, N - 1))))


if __name__ == "__main__":
    _selftest()
