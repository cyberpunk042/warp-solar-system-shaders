# `warp_shaders.blast`

A **research-grade nuclear-detonation model** — pure scaling-law functions that
size a fireball, blast rings, thermal radius, shock front and mushroom cloud from
a yield, plus the renderer that composites them over a landscape. Everything is
calibrated to the measured **Tsar Bomba** (50 Mt). See
[Research 15](../research/15-nuclear-fireball.md) for the physics + citations.

> ⚠️ These are declassified textbook civil-defence relations (Glasstone & Dolan,
> *The Effects of Nuclear Weapons*, 1977) — a visualisation of published physics,
> not weapon-design data.

```python
import warp_shaders as ws
ws.blast.TSAR.summary()
# 'Tsar Bomba: 50 Mt (~3846x Hiroshima) | fireball 3.5 km | destruction 38 km | thermal 100 km'
```

## Scaling laws (host)

All take a yield `w_kt` in **kilotons** and return **metres** (unless noted).

| Function | Returns | Law |
|---|---|---|
| `fireball_radius(w_kt)` | max fireball radius (m) | `46 · W^0.4` — 3.5 km at 50 Mt |
| `thermal_radius(w_kt)` | 3rd-degree-burn radius (m) | `1316 · W^0.4` — 100 km at 50 Mt |
| `destruction_radius(w_kt)` | total-destruction radius (m) | 5 psi contour, `1.03 km · W^⅓` |
| `severe_radius(w_kt)` | 20 psi contour (m) | `0.28 km · W^⅓` |
| `light_radius(w_kt)` | 1 psi contour (m) | `2.93 km · W^⅓` |
| `overpressure_radius(w_kt, psi)` | radius (m) of any overpressure | cube-root power fit |
| `shock_radius(t, w_kt, rho=1.2)` | Sedov–Taylor shock front (m) at time `t` | `1.03·(E t²/ρ)^(1/5)` |
| `mushroom_height(t, w_kt, tau=18)` | cap-top altitude (m) at time `t` | `H_max·(1−e^{−t/τ})`, 67 km ceiling |
| `fireball_temp(t_norm)` | effective blackbody temp (K) | 30,000 K → 1,500 K cooling |
| `debris_shell_radius(t, w_kt, m)` | vacuum plasma-shell radius (m) | ballistic `v·t`, `v=√(2E/m)` |

Blast effects scale by the **cube root** of yield (Sedov self-similarity); the
fireball and thermal radius scale by `W^0.4`. So a ×10 yield widens the blast
rings ×2.15 and the fireball ×2.5.

### `BlastParams`

Bundles the static radii for a named device. Prebuilt: **`TSAR`** (50 Mt) and
**`SUPER_TSAR`** (500 Mt).

```python
p = ws.blast.SUPER_TSAR
p.fireball, p.destruction, p.thermal, p.hiroshimas   # metres, metres, metres, ×
```

## Device helpers (`@wp.func`)

Call these inside your own kernel: `fireball_temp_at(core_k, r_norm)` (hot core →
cool rim), `blast_falloff(r, r_shock, width)` (Gaussian shell at the front),
`shock_ring(dist, ring_r, core_w, glow_w)` (a layered shockwave ring — sharp core
+ inner glow + soft outer glow — ported from
[`the-virus-block-mc`](https://github.com/cyberpunk042/the-virus-block-mc)'s
`shockwave_ring.glsl`), and `smoothstep(a, b, x)`.

## Renderer

`blast.render.render_ground(width, height, time, mouse, device, yield_kt)` — the
landscape + forest + volumetric fireball + mushroom + shock ring + tree damage,
sized from `yield_kt`. `blast.render.render_space(...)` — the vacuum variant (no
blast/fireball/mushroom; a ballistic plasma shell over a planet).
`blast.render.render_city(...)` / `render_suburb(...)` — **the nuke tested on a
built-up area**: a domain-repeated grid of `buildings` (towers/blocks for the city,
pitched-roof houses for the suburb) that **collapse into a burning field of rubble**
as the overpressure front sweeps out to the 5 psi `destruction_radius` — at dusk,
with lit windows extinguishing as each building comes down, smouldering embers +
thin rising smoke wisps in the flattened zone (see
[Research 18](../research/18-nuke-the-city.md)). Both wrap a shared `_render_blast`
with a `kind` flag. The [`tsar_bomba`](../gallery.md), [`super_tsar`](../gallery.md),
[`super_tsar_space`](../gallery.md), [`nuke_city`](../gallery.md) and
[`nuke_suburb`](../gallery.md) scenes are thin wrappers over these.

## Constants

`KT_J` (4.184×10¹² J/kt), `HIROSHIMA_KT` (13), `TSAR_KT` (5×10⁴),
`SUPER_TSAR_KT` (5×10⁵), `RHO_AIR` (1.2 kg/m³).
