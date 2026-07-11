# Research 14 — KIFS: folded fractal architecture

Sources and reasoning behind the second half of `warp_shaders/procedural/fractal.py`
— the **folding** fractals. Where the [Mandelbulb and Mandelbox](13-3d-fractals.md)
came from escape-time iteration, this family comes from **iterated function
systems (IFS)**: repeatedly *fold space onto itself* and scale toward a point.
Each fold is a mirror reflection; a handful of folds per step, iterated a dozen
times, carve out the Menger sponge, the Sierpinski tetrahedron, and — with a
rotation thrown into the loop — the endless cathedral interiors of a
**kaleidoscopic IFS (KIFS)**.

All three are still **distance-estimated**, so the engine's sphere-tracer draws
them exactly as it draws the Mandelbox: march by a lower bound on the distance to
the set, shade with the DE-gradient normal + soft shadows + AO + post.

## Folding space (Hvidtfeldt, "Syntopia" DE III)

The key idea ([Hvidtfeldt DE series III — "Folding Space"](http://blog.hvidtfeldts.net/index.php/2011/08/distance-estimated-3d-fractals-iii-folding-space/)):
a **conditional reflection** is distance-preserving, so you can build a fractal
by folding the *whole space* into one fundamental domain and evaluating a simple
primitive there. Two folds recur throughout:

- **plane fold** — reflect a point across a plane with unit normal `n` when it
  lands on the wrong side: `if dot(z, n) < 0: z -= 2·dot(z, n)·n`. A distance
  isometry (mirror), so it never distorts the DE.
- **abs fold** — the axis-aligned special case `z = |z|`, reflecting each octant
  into the positive one.

Because reflections preserve distance, a fold followed by a uniform **scale by
`s` about a fixed point `c`** (`z = s·(z − c) + c`) multiplies the running
derivative by `s` each iteration; the DE of the whole IFS is the DE of the final
primitive divided by that accumulated scale, `DE = d_primitive(z) / sⁿ`.

## The Sierpinski tetrahedron

The simplest 3D KIFS ([Hvidtfeldt DE III](http://blog.hvidtfeldts.net/index.php/2011/08/distance-estimated-3d-fractals-iii-folding-space/)):
three plane folds against the tetrahedron's mirror planes, then scale by 2 toward
a corner vertex `V`:

```
repeat n times:
    if z.x + z.y < 0: z.xy = -z.yx      # fold across x+y = 0
    if z.x + z.z < 0: z.xz = -z.zx
    if z.y + z.z < 0: z.yz = -z.zy
    z = 2·z - V·(2 - 1)                 # scale ×2 about V
DE = |z| · 2⁻ⁿ                          # distance to a point, unfolded
```

Each iteration replaces the tetrahedron with four half-size copies at its
corners — the 3D **Sierpinski gasket**. Twelve iterations already resolve finer
than a pixel.

## The Menger sponge (Quilez — an *exact* distance)

The Menger sponge is the 3D Cantor set: take a cube, drill a square hole through
each face, recurse into the 20 surviving sub-cubes. Inigo Quilez showed it has an
**exact** signed distance, not just an estimate
([Quilez — "menger fractal"](https://iquilezles.org/articles/mengersponge/)):
start from a box SDF, then each iteration **fold to the positive octant**
(`z = |z|`), sort so the two largest components are handled, scale by 3, and
**subtract a cross** (the drilled hole) sized to the current level:

```
d = sdBox(z, 1)
s = 1
repeat n times:
    a = mod(z·s·2, 2) − 1               # tile space at this level
    s *= 3
    r = 1 − 3·|a|
    da,db,dc = the three axis crosses of r
    c = (min(max(da,db), max(db,dc), max(dc,da)) − 1) / s
    d = max(d, c)                       # carve the hole
DE = d                                  # exact
```

Because it is a true SDF (signed, exact), the Menger sponge marches fast and
takes hard shadows cleanly — the sharpest of the three.

## Kaleidoscopic IFS — the temple (Knighty, 2010)

Add a **rotation** to the fold-and-scale loop and the self-similar copies no
longer line up on a grid — they spiral, and the fractal fills space with
column-and-arch "architecture." This is the **Kaleidoscopic IFS**, introduced by
the fractalforums user *Knighty* in 2010 and popularized through Hvidtfeldt's
**Fragmentarium**
([Hvidtfeldt — "Kaleidoscopic (escape time) IFS"](http://blog.hvidtfeldts.net/index.php/2011/08/distance-estimated-3d-fractals-iii-folding-space/),
[fractalforums KIFS thread](https://www.fractalforums.com/ifs-iterated-function-systems/kaleidoscopic-%28escape-time-ifs%29/)):

```
repeat n times:
    z = |z|                             # abs fold (octant mirror)
    rotate z about an axis by angle α   # the kaleidoscope
    z = scale·z − offset·(scale − 1)    # scale about the offset point
    (optional) rotate again
DE = |z| · scale⁻ⁿ
```

`offset` (the "Menger/Sierpinski" corner, ~`(1,1,1)`), `scale` (~2), and the two
rotation angles are the knobs — small changes sweep through cathedrals, lattices,
and coral. Animating a rotation angle makes the whole structure **breathe** and
reform, which is the scene's payoff.

## How it renders

`fractal.py` adds `menger_de(p, iters)`, `sierpinski_de(p, iters)`, and
`kifs_de(p, scale, angle, iters)` as device `@wp.func`s returning the DE plus the
same **orbit-trap** scalars (min `|z|`, final radius) the Mandelbulb/Mandelbox
scenes use for colour. The three scenes sphere-trace them, take the DE-gradient
normal, and shade with the engine's soft shadows + AO + sky + post — identical to
every other SDF scene, single-ray-per-pixel. The KIFS temple **rotates its fold
angle** over time so the architecture continually reassembles.

## Cross-references

- [Research 13 — Mandelbulb & Mandelbox](13-3d-fractals.md): the escape-time
  fractals and the orbit-trap colouring this shares; the Mandelbox's own folds
  are the bridge to this IFS family.
- [Research 00 — foundations](00-foundations.md): the sphere-tracing raymarch +
  SDF gradient normal reused here.
- Folding + KIFS: [Hvidtfeldt "Syntopia" DE series](http://blog.hvidtfeldts.net/index.php/category/distance-estimation/)
  (parts III–IV), [Knighty's Kaleidoscopic IFS](https://www.fractalforums.com/ifs-iterated-function-systems/kaleidoscopic-%28escape-time-ifs%29/).
- Menger sponge exact distance: [Quilez](https://iquilezles.org/articles/mengersponge/).
- The fractals: [Menger sponge](https://en.wikipedia.org/wiki/Menger_sponge),
  [Sierpiński tetrahedron](https://en.wikipedia.org/wiki/Sierpi%C5%84ski_triangle#Analogues_in_higher_dimensions).
