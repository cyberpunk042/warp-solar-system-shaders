# Research 02 — Atmospheric scattering (single + multiple)

Sources and derivations behind `warp_shaders/engine/atmosphere.py`.

## Single scattering (the base)

The sky's color is sunlight scattered toward the eye by air molecules (Rayleigh)
and aerosols (Mie). We march the view ray; at each sample we accumulate in-
scattered Rayleigh + Mie weighted by their phase functions and the Beer–Lambert
transmittance along both the view path and the sun path, with a ground-
intersection test for the terminator/night side.

- **Rayleigh** — wavelength-dependent (`β_R = (5.8, 13.5, 33.1)×10⁻⁶` per m),
  phase `3/16π · (1 + µ²)`. This is why the zenith is blue and the low sun reddens.
- **Mie** — grey (`β_M = 21×10⁻⁶`), Cornette–Shanks phase with `g = 0.76` — the
  bright forward halo around the sun and the white horizon haze.
- Earth-like scale heights `H_R = 8 km`, `H_M = 1.2 km`; radii `R_g = 6360 km`,
  `R_a = 6420 km` (SI metres in code).

Sources: Nishita et al., *Display of the Earth Taking into Account Atmospheric
Scattering* (SIGGRAPH 1993); Sean O'Neil, *Accurate Atmospheric Scattering*
(GPU Gems 2, ch. 16); Cornette & Shanks (1992) for the Mie phase.

### Transmittance LUT (acceleration)

The inner sun light-march (optical depth from each view sample to the sun) is the
same function of `(altitude, sun-zenith µ)` every frame, so we precompute it once
into a 2D LUT (`bake_transmittance` → `build_transmittance_lut`) and read it with
`transmittance_lut(lut, h, µ)`. The LUT also bakes sun **visibility** (0 when the
planet occludes the sun). Resolution scales with the quality tier
(`QualityTier.lut_size`). Pattern from Bruneton & Neyret, *Precomputed Atmospheric
Scattering* (2008).

## Multiple scattering (Hillaire 2020)

Single scattering alone leaves the sky too dark away from the sun and at twilight,
because in reality light scatters **many** times before reaching the eye. Rather
than march high-order paths (prohibitive), we use Sébastien Hillaire's insight
(*A Scalable and Production Ready Sky and Atmosphere Rendering Technique*, EGSR
2020): approximate all orders ≥ 2 as **isotropic** and sum them as a geometric
series.

For each `(altitude, sun-µ)` texel we integrate over a deterministic
**Fibonacci-sphere** direction set (no RNG — Warp forbids it) to estimate:

- `L₂` — the 2nd-order in-scattered luminance (single-scattered light arriving
  from every direction, read from the transmittance LUT), and
- `f_ms` — the fraction a unit ambient re-scatters (the transfer factor).

Then `ψ_ms = L₂ / (1 − f_ms)` sums orders 2, 3, 4, … in closed form. Because the
multiscatter phase is isotropic (`1/4π`), the `4π` solid angle and `1/4π` phase
cancel, so the sphere integral is just the average over directions. The bake
(`bake_multiscatter` → `build_multiscatter_lut(tr_lut, …)`) consumes the
transmittance LUT and produces a second LUT sampled by `multiscatter_lut`.

At runtime `atmosphere_lut` adds, per view sample,
`ψ_ms · σ_s(h) · view_transmittance` — **isotropic, no sun phase, no inner loop** —
so it lifts the shadowed lower sky, horizon, and blue hour while leaving the
sun-side single-scatter gradient intact.

### Calibration note

This engine's single-scatter is artistically boosted (arbitrary `_SUN_I = 22`) to
look right on its own, so a full-strength physical multiscatter term would double-
count and blow out midday. `_MS_SCALE = 0.22` folds the Hillaire term into the
model's tuned radiance budget: a measured ~+7 % overall lift at low sun with a
richer, bluer upper sky, and midday essentially unchanged. The term is physically
derived; the scale places it correctly within a non-radiometric model.

Verified by rendering `scenes/sky.py` at low sun with the multiscatter LUT vs a
zeroed LUT (single-scatter only): the upper-sky blue channel rises 0.887 → 0.953.
