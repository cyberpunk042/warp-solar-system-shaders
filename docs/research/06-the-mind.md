# Research 06 — the mind: choosing to obey the obvious rules

The top of the "show life" ladder. The plants already obey the **obvious rules**
reflexively — a tropism *always* bends them toward light, rain *always* folds
their leaves ([Research 04](04-lsystems.md) §2.3.4). A **mind** is what lets a
plant *choose*: to follow the light only when it is worth it, to close up, to
rest. Source behind `warp_shaders/life/mind.py` and `scenes/mind.py`.

## Why Conway's Game of Life

The operator framed the mind as *"L-Systems with a Conway Game of Life / a
Mind."* Conway's Life (Martin Gardner, *Scientific American*, 1970) is the
canonical example of **complex behaviour emerging from trivial local rules**: a
cell lives or dies from its eight neighbours (born on exactly 3 neighbours,
survives on 2 or 3). It is **Turing-complete** — patterns in Life can compute
anything computable — so it is a legitimate, minimal substrate for "a
deliberation." That is exactly the shape a mind needs: many tiny local
interactions summing to a global choice.

`Mind` runs a toroidal Life grid and reads a single scalar off it — the **living
population** — smoothstepped into a **drive** in [0, 1]. A periodic *stimulus*
(a small random burst — a sensory poke) keeps the deliberation from dying out, so
the drive waxes and wanes as the little mind churns. Everything is deterministic
from `seed`: the same seed replays the same train of thought.

## Choice = overriding the obvious rule

The scene maps the drive onto the very same tropism knobs the reflex scenes use —
but now as a **decision**, not a reflex:

| Mind state | drive | plant does |
|---|---|---|
| active (busy grid) | high | **seeks the light** — strong phototropism, leaves open |
| quiescent (sparse grid) | low | **rests** — sags under gravity, leaves folded shut |

So where `phototropism` *always* follows the light, `mind` follows it **only when
the mind decides to**, and closes up when it doesn't. The CA is drawn as an inset
panel with a drive bar, so the deliberation and its consequence are visible at
once. This is the operator's *"the mind can choose to follow the light or to
close pieces of itself when it rains"* — the reflex is the actuator, the mind is
the will.

## The seam

The mind steers the plant through the exact hook the "obvious rules" were built
on: it constructs a per-frame `TurtleConfig` (light gain, gravity gain, leaf
fold) from `Mind.decision()` and calls `plants.grow_mesh_env(spec, gen, cfg)`, so
the *same* grammar re-expresses under the mind's command every frame. Today the
decision is **whole-plant**; the context-sensitive L-System class (a decision
symbol propagating through the structure, [Research 04](04-lsystems.md)) is the
substrate for a future **per-branch** mind — one shoot chasing the light while
another rests.

## The conceptual horizon (operator framing)

The operator situated the mind in a larger idea: that a mind's influence reaches
*"'backward' in time, in the snapshot / visual part of time like 1/3000000 …
those not being seconds but a 'real' timescale to explain how things are waves
before what to us seems like a collapse in the world."* That is a
wave-function-collapse metaphor for deliberation — the mind exploring
superposed possibilities on a timescale far finer than the visible frame, the
observed plant being the "collapsed" outcome. This engine records that as the
**horizon**, not the implementation: the mind's first *realized* influence here
is the **choice** — override or obey. The finer-timescale, retrocausal framing is
noted for where this strand goes next.

## What comes next

Up from here: a **per-branch** mind (context-sensitive signals steering
individual shoots), competition between plants in the `meadow`, and the
wave/collapse timescale above. Down the ladder is already built — DNA → protein →
cell ([Research 05](05-molecular-to-cell.md)) — so the engine now shows life from
molecule to mind.
