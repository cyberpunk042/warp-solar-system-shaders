"""warp_compress — a folding, chromosome-coiling compressor.

The idea, made literal from two metaphors:

* **Fold the card into a cube.** A flat byte/token stream is reshaped into an n-D lattice and
  read back along a space-filling curve (Morton / Hilbert). Folding is a *reversible reordering*
  that pulls symbols which are far apart in the stream but near in the fold into adjacency, so
  repetition that a linear scan would miss surfaces for the next stage. (Geometric cousin of the
  Burrows–Wheeler transform.)

* **Wrap it layer by layer into a chromosome, like DNA.** Hierarchical grammar coiling (Re-Pair):
  the most frequent adjacent pair of symbols is wrapped into a new unit — a *nucleosome* — then
  nucleosomes coil into fibers, fibers into loops, forming a compact *chromosome* = a reusable
  rule dictionary (the histone scaffold) plus a short top-level strand (the chromatid). Expanding
  the grammar is exact decompression.

Two modes:

* **lossless** — fold + coil only; ``decompress(compress(x)) == x``.
* **lossy** — quantize first, so near-identical motifs collapse onto the *same* nucleosome and the
  strand coils far tighter, at a controlled error; decompression reconstructs the quantized strand.

The north star (future): symbols become word-IDs, nucleosomes become phrases, higher coils become
idioms → sentences → concepts, and a semantic lossy tier folds near-synonyms onto one symbol —
compressing *meaning*, not just bytes.
"""

from . import fold  # noqa: F401  (submodule: warp_compress.fold.fold / .unfold)
from .chromosome import Chromosome, coil, uncoil
from .codec import compress, decompress, describe

__all__ = [
    "compress",
    "decompress",
    "describe",
    "coil",
    "uncoil",
    "Chromosome",
    "fold",
]
