# Research 03 — Volumetrics, cloud detail, and god-rays

Sources behind `engine/volumetric.py`, the baked cloud-detail volume, and the
`post.godrays` light-shaft pass (used by `sky`, `canyon`, `reef`).

## Volumetric clouds

`march_clouds` raymarches a horizontal density slab: at each step it accumulates
in-scattered sunlight with **Beer–Lambert** extinction and a short **light-march**
toward the sun for self-shadowing, weighted by the **Henyey–Greenstein** phase
function and a "powder" term for the dark-edge look. Density (`cloud_density`)
follows the Schneider "Nubis" recipe — a low-frequency fBm shape, a coverage
remap, a height gradient (rounded bottoms, soft tops), and high-frequency edge
erosion.

Sources: Andrew Schneider & Nathan Vos, *The Real-time Volumetric Cloudscapes of
Horizon Zero Dawn* (SIGGRAPH 2015); Henyey & Greenstein (1941); Beer–Lambert.

### Baked detail volume

The high-frequency erosion field is the same seamless 3D function at every march
step, so recomputing Worley noise per step is wasteful. `build_cloud_detail`
bakes it **once** into a `wp.array3d` (a tileable fBm of `value_tiled3`), and
`cloud_density` reads it with a single trilinear `sample3d` fetch (wrap-repeat).
This replaces a 27-tap Worley evaluation per step with one texture read —
**~45 % faster** on the `clouds` scene at equal visual detail — and stays
seamless because `value_tiled3` is periodic on the bake lattice (see
[Research 01](01-textures-and-luts.md)). The spherical cloud shell in `earth_v2`
keeps its own direction-space Worley erosion (a flat tile doesn't map cleanly to
a sphere).

## God-rays (crepuscular shafts)

`post.godrays` is a screen-space **radial light-scattering** pass (Mitchell,
*Volumetric Light Scattering as a Post-Process*, GPU Gems 3, 2007). It threshold-
masks the bright pixels (the sky gap / sun), then marches each pixel toward the
light's screen position accumulating that bright image with exponential decay —
producing shafts that appear to emanate from the light.

Because it's screen-space, a scene only needs to (1) render a bright light region
in frame and (2) project the light's world direction to screen coordinates
(`cx, cy`) before calling `godrays`. This engine uses it in three very different
settings:

- **sky** — the sun disk seeds shafts through the atmospheric haze.
- **canyon** — the sun sits in the slot's sky gap; shafts pour down between the
  sandstone walls (the classic Antelope-Canyon light beam).
- **reef** — a near-surface sun throws shafts down through the water onto the
  seabed, over the Worley **caustic** network on the sand.

## Caustics (reef)

Underwater caustics — the dancing bright web focused light casts on the seabed —
are approximated with **Voronoi (Worley) edge bands**: `worley3_f2` returns the
two nearest feature distances `(F1, F2)`, and `F2 − F1` is small along cell
boundaries, so `pow(clamp(1 − k·(F2−F1)), 2)` lights a moving network. Two
drifting octaves at different scales give the layered shimmer. This is a
stylized, cheap stand-in for true refractive caustics (which would trace light
through an animated water surface) but reads convincingly in motion.
