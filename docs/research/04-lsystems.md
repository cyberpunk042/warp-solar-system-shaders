# Research 04 — L-Systems: growing life

Sources behind `warp_shaders/life/` — the engine growing plants from grammars.

## What an L-System is

A **Lindenmayer system** rewrites a string of symbols (*modules*) **in parallel**
— every module at once — over discrete generations. That parallelism is the
whole point: it's the mathematics of cells dividing simultaneously, which is why
L-Systems model plant growth so naturally. Aristid Lindenmayer introduced them in
1968 to describe the development of filamentous organisms; the canonical
reference is Przemysław Prusinkiewicz & Aristid Lindenmayer, *The Algorithmic
Beauty of Plants* (Springer, 1990) — "ABOP".

The simplest example (Lindenmayer's algae): axiom `A`, rules `A → AB`, `B → A`.
Successive generations are `A`, `AB`, `ABA`, `ABAAB`, `ABAABABA`, … — and their
**lengths are the Fibonacci numbers**.

## The classes (`life/lsystem.py`)

One uniform `Rule` covers all four, selected by match gates:

- **D0L** — deterministic, context-free: one successor per symbol (Koch curve,
  dragon curve, the algae above).
- **Stochastic** — several successors with weights; a seeded RNG chooses one,
  reproducibly, so a field of "the same" plant varies naturally (ABOP §1.7).
- **Context-sensitive** (IL / 1L / 2L) — a rule fires only with the required left
  / right neighbour. This is how *signals* propagate through a structure (a
  hormone travelling up a stem) — and the hook the future "mind" layer will use.
  Matching is bracket-aware and skips an `ignore` set (ABOP §1.8).
- **Parametric** — modules carry numeric parameters and rules gate on a condition
  and compute successor parameters with arithmetic, e.g. `A(ℓ,w) → !(w)F(ℓ)[…]`
  with `ℓ, w` shrinking each level. This is what makes a tree *taper* (ABOP §1.10).

## Turtle interpretation (`life/turtle.py`)

The rewritten word is drawn by a **3D turtle** carrying an orthonormal
**H / L / U** frame (heading / left / up). `F` extrudes a branch segment; `+ - &
^ \ /` rotate the frame (yaw / pitch / roll); `[` and `]` push/pop the whole
state, which is what turns a linear string into a *branching* structure (ABOP
§1.6). `!` sets branch width, `'` the colour, `L` drops a leaf. This is the
bracketed-turtle interpretation, in 3D.

## Rendering — the engine shows life (`life/mesh.py`, `life/render.py`)

Turtle output is tessellated to a triangle mesh — each segment a tapered tube,
each leaf a blade — and uploaded as a `wp.Mesh`, which builds a **BVH**. A Warp
kernel then casts one camera ray per pixel with `wp.mesh_query_ray`, barycentric-
interpolates the vertex normal + colour, and shades with the engine's GGX PBR,
sun, sky, cast shadow (a second ray toward the sun), and post pipeline. So the
plants are **real generated 3D geometry, ray-traced** — not sprites.

## The three plants (`life/plants.py`)

Increasing complexity, the start of the DNA → cell → grass → plant → tree arc:

- **grass** — a tuft of arching blades (each blade an internode chain pitched a
  little every generation).
- **herb** — a stochastic bracketed plant with leafy side-branches placed in
  **137.5° golden-angle phyllotaxis** (the real angle at which many plants space
  successive leaves).
- **tree** — a parametric ternary 3D tree: a tapering woody trunk that splits
  three ways per level, twigs leafing out once their width drops below a
  threshold.

Growth is simply deriving to a higher generation: advancing `time` grows the
plant one generation, so it rises into frame from a sprout.

## What comes next

The roadmap continues down (DNA → protein → cell, the sub-plant scales) and then
adds a **mind**: a decision layer (Conway-style local rules maturing into
choice) that overrides the plant's default growth — turning toward light, closing
in rain — and, per the operator's framing, reaches *backward* through a
wave-collapse timescale. The context-sensitive class above is the substrate for
those signals.
