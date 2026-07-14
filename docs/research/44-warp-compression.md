# Research 44 — Warp compression (folding *is* the compression)

> A new invention: **warp** = fold. Compress data by *folding* it — a card folded into a cube, a
> strand wrapped into a chromosome — so that where the folded layers land on each other and
> **match**, they merge and are stored once. Lossless when we keep the differences, lossy when we
> drop the small ones. Reversible: unfolding restores the original. And because folding is a
> *process*, it is rendered **in time** — you watch the fold happen and unwind, for more than one
> kind of data. The north star is *words and meaning*; the mechanism is built and verified on
> bytes and cards first.

## Folding the actual card (the RTX board → cube / chromosome)

The headline scenes fold the **real graphics card**. `gpu_board` (the RTX 6000 Pro Blackwell board)
is a raymarched SDF; warp compression folds its **domain**: the long card is accordion-folded in x
and z and layer-stacked in y, then the folded material is bounded into a shape — a **mini-cube**
(`warp_fold_card`) or the **X of a chromosome** (`warp_fold_chromo`). The board's own materials —
green solder mask, gold routing, GDDR7, the die floorplan — fold and stack with it, so the whole
card condenses into a compact glowing form built from its own silicon and copper, then unfolds flat
again. `time` drives compress → decompress. This is the fold, applied to a thing you recognise.

## The one idea, two shapes (the underlying byte/card codec)

Folding brings distant parts of the data into **contact**. Wherever the parts that touch are the
same, one copy suffices — that is the compression. Do it in layers and the data condenses into a
small, dense form; replay the differences and it unfolds back. Two shapes, one mechanism:

### Strand → chromosome (`warp_compress.wrapfold`)

Take a 1-D strand and **wrap it onto a cylinder** whose circumference is the period the codec
finds (the shift `p` whose coil-on-coil agreement is highest). Now each coil lands on the one
below it. Look down each column: wherever a cell equals the cell in the coil beneath, it **merges**
— kept once, recorded as a single "same" bit; only the differences store a value. Then the template
coil is itself wrapped and merged — layer by layer — the strand condensing into a **chromosome** of
nested coils. Unwrapping replays the differences from the innermost core outward, exactly.

It finds real structure on its own: motif length 4 for `ACGTACGT…`, the phrase length for repeated
text, the wave period for a sinusoid; noisy data folds in several nested coils (e.g. periods
112 → 36 → 16 → 4). Verified exact lossless round-trip on periodic / DNA / text / sine / random /
empty inputs; DNA compresses ~6×.

### Card → cube (`warp_compress.cardfold`)

Take a 2-D **card** and fold it in half like paper — the far half mirrors onto the near half.
Where the two stacked cells match, they **merge**; where they differ, the difference is kept. Fold
again on the other axis, and again, alternating — the card halving and thickening into layers,
condensing toward a compact **cube** whose bright core is the fundamental tile the whole card was
built from. Unfolding replays the differences outward.

A card with mirror symmetry at several scales folds all the way down: a 64×64 self-similar card
folds **8 times** to a 4×4 core, exact. Verified round-trip on symmetric / random / flat cards.

## Two modes (both shapes)

| Mode | What it does | Guarantee |
|---|---|---|
| **lossless** | fold + merge only (`tolerance = 0`) | `decompress(compress(x)) == x` exactly |
| **lossy** | merge cells that agree **within `tolerance`**, dropping the small difference | reconstruction error ≤ `tolerance` per cell |

The lossy dial is a genuine rate–distortion knob: a bigger tolerance merges more of the fold, so
the blob shrinks at bounded, graceful error (measured: a noisy periodic signal, lossless 2253 →
lossy 227 bytes within the bound).

## Seeing it happen — folding in time

A compressor is usually shown before/after. Folding is a *process*, so the engine renders the real
fold schedule as it runs — the fold *and* the unfold:

- **`warp_card`** — a flat card of glowing cells folds in half again and again; matching cells flash
  green and merge into the layer below, condensing into a bright cube core, then unfolds flat.
- **`warp_fold`** / **`warp_fold_words`** — a strand wraps onto the period the codec found; cells
  that match the coil below flash green and telescope up into one bright template ring (the
  repeating unit), then unwrap.

Both replay one full cycle: compression, then decompression. The frame *is* the fold step — what
you see is what the codec did.

## Where it stands (honest)

Lossless round-trips are exact on every input tested, and folding genuinely compresses self-similar
data. It does **not yet beat gzip** on general data, for one clear reason: the fold's merge-bitmaps
and stored differences are packed with plain LEB128 varints — there is no entropy coder on that
stream. gzip's tuned LZ77 + Huffman still wins the last factor. That factor is exactly the headroom
and is a *next* stage, not a flaw in the fold-and-merge idea.

## The north star — words and meaning

Everything is alphabet-agnostic. Point the strand fold at **word-IDs** instead of bytes and the
chromosome becomes linguistic: coils are phrases, matching coils are repeated phrases that merge,
nested layers are idioms → clauses → concepts — "so much words and meaning wrapped layer by layer,
like a DNA strand." A **semantic lossy tier** would merge near-synonyms (agree "within tolerance"
in meaning) before folding — compressing the *sense* while dropping only surface form. That is the
next round; the fold-and-merge machine it rides on is built and verified here.

## Sources & lineage

- M. Burrows & D. Wheeler (1994) — reorder-to-expose-structure, the spirit of folding.
- N. J. Larsson & A. Moffat, *"Offline dictionary-based compression"* (Re-Pair), Proc. IEEE 88
  (2000) — hierarchical merging of repeats (the alternate coil in `warp_compress.chromosome`).
- K. Luger et al., *"Crystal structure of the nucleosome core particle"*, Nature 389 (1997) — the
  DNA layer-wrapping the chromosome metaphor borrows.
- Origami / paper-folding as a model of hierarchical spatial self-similarity.
