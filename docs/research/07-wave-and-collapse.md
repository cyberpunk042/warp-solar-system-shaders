# Research 07 — wave and collapse: the plant's futures resolving to one

The distinctive idea at the top of the "show life" ladder, in the operator's own
framing:

> *"the mind will be in the next development and will have even more impact
> 'backward' in time, in the snapshot / visual part of time like 1/3000000 (or
> 1/3000 or 1/3 depending on the scale)… those not being seconds obviously but a
> 'real' timescale to explain like in real life how things are waves before what
> to us seems like a collapse in the world."*

Source behind `warp_shaders/life/collapse.py` and `scenes/wavecollapse.py`.

## The metaphor (explicitly a metaphor)

This is a **visual metaphor**, not literal quantum mechanics. The analogy it
borrows:

| Physics (measurement) | Here (a growing plant) |
|---|---|
| wavefunction — a superposition of possible states | an **ensemble** of possible plant futures (different grammars / seeds) |
| the state before measurement — a "wave" | the **superposition image**: the futures rendered faint and overlapping — a cloud of what the plant *might* become |
| measurement / decoherence | the **collapse**: the cloud resolves to a single realised plant |
| what biases the outcome | a Conway **mind** picks *which* future resolves |

The engine does not simulate amplitudes or Born-rule probabilities — it renders
an ensemble, superposes the images (their mean is the "wave"), and blends toward
one member (the "collapse"). The value is the *picture* of the idea: possibility
becoming actuality.

## Backward in time — the collapse front

The operator's "'backward' in time" is the striking part. The collapse is not a
global fade; it sweeps a **front** through the plant from the **tips downward to
the base**. The tips are the plant's *future* (the last thing it grows); the base
is its *past* (grown first). So the realised form crystallises **future-first**,
the collapse reaching *backward* through the plant's own growth history — the
settled future retroactively fixing what the earlier structure "was."
`collapse_blend` implements this as a per-row alpha: rows above the front show the
chosen plant, rows below stay superposed, and the front descends over `time`
across the subject's pixel extent.

The `1/3000000 … 1/3 depending on the scale` is the operator's point that this
"snapshot" timescale is *relative* — the wave/collapse happens far finer than the
visible frame, and what we see is only the collapsed outcome. The engine records
that as the framing; what it renders is the collapse itself, slowed to something
the eye can follow.

## How it is built (reuse, not reinvention)

- **Ensemble** — five visibly-distinct futures (herb, bush, flower, fern,
  sapling), all rooted at the same spot, grown with the existing L-System plants.
- **Render once, blend many** — the camera is fixed, so the five plants are
  ray-cast **once** (through the same `render_plant` the whole life strand uses)
  and cached; only the cheap per-frame image blend re-runs.
- **Superposition** — `superpose` is the (weighted) mean of the ensemble images:
  the faint overlapping ghost cloud.
- **Collapse** — `collapse_blend` sweeps the front tip→base, resolving to the
  chosen member.
- **The mind chooses** — the Conway `Mind` (from [Research 06](06-the-mind.md))
  supplies `decisions(k)`; `pick_index` takes the strongest band, so *which*
  future collapses is the mind's — fixed at a decision moment for a stable
  outcome, while the grid keeps evolving in the inset for flavour.

## Where this sits

This is the conceptual summit of the strand: molecule → cell → plant → mind →
**the mind biasing which future becomes real**. What remains open (per the
operator's framing) is finer structure — per-*node* superposition (a single
branch's tip in several possible states), amplitude-weighted rather than
uniform ensembles, and the interplay of many minds. The engine now shows life
from a strand of DNA to a wave of possible futures collapsing under a mind's
choice.
