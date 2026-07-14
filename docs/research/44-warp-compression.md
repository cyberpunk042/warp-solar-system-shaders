# Research 44 — Warp compression (fold the card, coil the chromosome)

> A new invention, separate from the renderer but sharing its spirit: compress data by **folding**
> it into a lattice and **coiling** it layer by layer into a chromosome, the way DNA packs metres
> of strand into a nucleus. Lossless when you only fold and coil; lossy when you quantise first.
> The north star is *words and meaning* — but the mechanism is built and verified on bytes first.

## The two metaphors, made literal

**Fold the card into a cube.** A flat stream is laid row-major into a square (2-D) or cube (3-D)
and read back along a **space-filling curve** (Morton / Z-order here; Hilbert next). A fold is a
*reversible permutation* — it compresses nothing by itself, it *rearranges* so that symbols which
are far apart in the line but near in the fold become adjacent, surfacing repetition a linear scan
would miss. This is the geometric cousin of the Burrows–Wheeler transform: reorder to expose
structure, then let the next stage exploit it. Every fold is exactly invertible, so the encoder is
free to **search** folds and keep whichever coils smallest.

**Coil it like a chromosome.** The reordered strand is compressed by hierarchical grammar coiling
(**Re-Pair**). Each pass wraps the most frequent adjacent pair of symbols into a fresh symbol and
records a rule `new -> (a, b)` — one **nucleosome**, a pair wrapped into a bead. Because a rule's
members can be earlier rules, the wrapping compounds: nucleosomes coil into fibers, fibers into
loops. What remains is a compact **chromosome**:

- a **rule dictionary** — the reusable *histone scaffold* (motifs shared across the whole strand);
- a short **top strand** — the *chromatid* (what's left after all the coiling).

Uncoiling expands every rule back to its two members until only literals remain — exact, lossless
decompression. The biology is a faithful analogy: DNA achieves ~10,000× linear compaction by
exactly this trick — repeated units (histones) wrapped in structured, hierarchical layers.

## The two modes

| Mode | What it does | Guarantee |
|---|---|---|
| **lossless** | fold + coil only | `decompress(compress(x)) == x` exactly |
| **lossy** | **quantise** to `q`-wide levels *before* folding, so near-identical motifs collapse onto the *same* nucleosome and the strand coils far tighter | reconstruction error ≤ `q/2` per symbol |

The lossy step is a genuine **rate–distortion dial**: a bigger `q` shrinks the alphabet (fewer
distinct nucleotides), which multiplies the coiling, which shrinks the blob — at graceful, bounded
error. Measured on a noisy sinusoid (2,500 samples), lossless can't shrink it at all (it is
incompressible symbol-for-symbol), but the lossy dial does:

| `q` | alphabet | packed bytes | mean \|error\| |
|---|---|---|---|
| 1 (lossless) | 256 | 4671 | 0.00 |
| 8 | 32 | 1900 | 1.98 |
| 16 | 16 | 1171 | 4.05 |
| 32 | 8 | 680 | 8.92 |

## Where it stands (honest)

Lossless round-trips are exact on every input tested (repetitive text, DNA-like motif strings,
source code, 2-D gradients, empty/single/random bytes). On *repetitive* data it compresses hard
(≈55× on `"the quick brown fox " × 300`). It does **not yet beat gzip** on general data, for one
clear reason: the coiled grammar is currently packed with plain LEB128 varints — there is no
entropy coder on the rule stream. gzip's tuned LZ77 + Huffman still wins the last factor of ~2.
That factor is exactly the headroom, and it is squarely a *future* stage, not a flaw in the fold /
coil idea:

| next step | what it buys |
|---|---|
| **entropy-code the chromosome** (range/arithmetic coder over rules + top) | the missing ~2× vs gzip; makes lossless competitive |
| **richer folds** (Hilbert 2-D/3-D, learned windings) + smarter search | pays off on data with 2-D/3-D locality that row-major separates |
| **the word layer** (below) | the real prize |

## The north star — words and meaning

Everything above is alphabet-agnostic. Point it at **word-IDs** instead of bytes and the chromosome
becomes linguistic:

- **nucleotides → words**; **nucleosomes → frequent phrases** (collocations); higher coils →
  **idioms → clauses → sentences → concepts**. The histone scaffold becomes a shared phrase-book
  the whole corpus reuses.
- a **semantic lossy tier**: cluster near-synonyms onto one symbol *before* coiling — compressing
  the *meaning* while discarding only surface form. Lossy on words, ~lossless on sense. This is the
  literal reading of "wrapping so much words and meaning layer by layer, forming the chromosome."

## Sources & lineage

- N. J. Larsson & A. Moffat, *"Offline dictionary-based compression"* (Re-Pair), Proc. IEEE 88
  (2000) — the grammar-coiling core.
- C. Nevill-Manning & I. Witten, *"Identifying hierarchical structure in sequences"* (Sequitur),
  JAIR 7 (1997) — hierarchical grammar induction.
- M. Burrows & D. Wheeler (1994) — reorder-to-expose-structure, the spirit of the fold.
- G. M. Morton (1966) — Z-order space-filling curve; D. Hilbert (1891) — the Hilbert curve.
- K. Luger et al., *"Crystal structure of the nucleosome core particle"*, Nature 389 (1997) —
  the DNA-wrapping biology the metaphor borrows.
