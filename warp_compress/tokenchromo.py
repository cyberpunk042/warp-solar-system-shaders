"""C3 — tokenize -> web -> DNA -> chromosome: the card as a genome, coiled (lossless).

Operator spec (2026-07-14, verbatim): *"the break down of the item into a web of word that remesent per
each atom a word, or a token rather in a web that gives values and we can then compress it from DNA
equivalent sequence into the whole process of chromosome."*

Break the card down so **each atom becomes a token**; the tokens form a **web that gives values**
(the vocabulary: token -> the piece it stands for); reading them in order is the **DNA-equivalent
sequence** (a genome of the card); and that sequence is compressed through the **whole chromosome
process** — the Re-Pair coil (`warp_compress/chromosome.py`), where repeated token-*phrases* wrap into
nucleosome rules, layer by layer, exactly like a chromatin fibre condensing into a chromosome.

Composition: the **atom -> token** step is the C1 dedup (each unique block is a word); C3 adds the
**sequence-level** coil on top, so repeats that C1's flat index could not exploit (a whole *row* of
identical memory, a repeated VRM motif) merge as phrases. Lossless end to end: uncoil the chromosome
-> the DNA sequence -> place the vocabulary pieces back -> the exact card. Verified in
``tests/test_tokenchromo.py``.
"""

import numpy as np

from . import mergecube as mc
from .chromosome import coil, uncoil
from .foldcube import sample_card


def tokenize(occ, block=4):
    """Break the card into atoms and give each a token (word). Returns the vocabulary (web of values),
    the index shape, the codec meta, and the DNA-equivalent token sequence (scan order)."""
    vocab, index, meta = mc.compress(occ, block=block)       # unique pieces = the words
    seq = index.reshape(-1).astype(np.int64).tolist()        # the genome: one token per atom, in order
    return vocab, index.shape, meta, seq


def untokenize(vocab, index_shape, meta, seq):
    index = np.array(seq, np.int32).reshape(index_shape)
    return mc.decompress(vocab, index, meta)


def _bits(n_symbols, alphabet):
    return n_symbols * int(np.ceil(np.log2(max(alphabet, 2))))


def compress_card(block=4):
    """Sample the real board, tokenize -> coil into a chromosome, verify lossless, measure."""
    occ = sample_card()
    vocab, ishape, meta, seq = tokenize(occ, block=block)
    n_unique = len(vocab)

    chrom = coil(seq, base=n_unique)                         # DNA -> chromosome (Re-Pair coil)
    seq_back = uncoil(chrom)
    card_back = untokenize(vocab, ishape, meta, seq_back)

    lossless = bool(np.array_equal(card_back, occ) and seq_back == seq)

    # sizes (bits): the vocabulary "web" + the coiled chromosome (rules + top strand)
    orig_bits = int(occ.size)
    vocab_bits = int(vocab.size)                             # the unique pieces, stored once
    n_rules = len(chrom.rules)
    alphabet = n_unique + n_rules
    chrom_bits = _bits(2 * n_rules + len(chrom.top), alphabet)   # each rule = a pair; + top strand
    comp_bits = vocab_bits + chrom_bits

    # for contrast: C1 alone would store the flat index at fixed width
    c1_bits = vocab_bits + _bits(len(seq), n_unique)

    st = chrom.stats()
    return {
        "lossless": lossless,
        "atoms": len(seq),                  # DNA length (one token per atom)
        "vocab": n_unique,                  # words in the web
        "nucleosomes": st["nucleosomes"],   # coil rules (merged token-phrases)
        "layers": st["layers"],             # coil depth (chromatin condensation level)
        "top_symbols": st["top_symbols"],
        "ratio": orig_bits / max(comp_bits, 1),
        "ratio_vs_c1": c1_bits / max(comp_bits, 1),   # how much the chromosome coil beats flat C1
        "block": block,
    }
