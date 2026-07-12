# Research 15 — the nuclear fireball: Tsar Bomba, scaling laws, and vacuum

Sources and reasoning behind `warp_shaders/blast/` — a **physically-grounded**
nuclear-detonation model built to real scaling laws, driving the `tsar_bomba`,
`super_tsar`, and `super_tsar_space` scenes. The goal is a life-like simulation:
the fireball, the blast wave, the mushroom cloud, and the damage rings are all
sized from the yield by the same laws used in *The Effects of Nuclear Weapons*
(Glasstone & Dolan, 1977), calibrated against the measured Tsar Bomba test.

> ⚠️ This is a **visualisation of published, declassified physics** (fireball
> radius, blast overpressure, thermal fluence, cloud rise). It is not weapon-design
> information — the scaling laws below are textbook civil-defence relations.

## The two devices

| | Yield | × Hiroshima¹ | Energy |
|---|---|---|---|
| **Tsar Bomba** (RDS-220, tested 1961) | **50 Mt** (5×10⁷ kt) | ~3,800× | 2.1×10²⁰ J |
| **Super Tsar** (hypothetical ×10) | **500 Mt** (5×10⁸ kt) | ~38,000× | 2.1×10²¹ J |

¹ Hiroshima ("Little Boy") ≈ 13 kt, so 50 Mt ≈ 3,800×
([Wikipedia — Tsar Bomba](https://en.wikipedia.org/wiki/Tsar_Bomba)). The Tsar
Bomba was **deliberately halved** from its 100-Mt design to limit fallout; the
Super Tsar here takes the tested 50-Mt device and scales the *yield* ×10 to show
how the effect radii grow — a controlled numerical experiment.

## Measured anchors (Tsar Bomba, 50 Mt)

The whole model is calibrated so that at 50 Mt it reproduces the observed test
([Wikipedia](https://en.wikipedia.org/wiki/Tsar_Bomba),
[National WWII Museum](https://www.nationalww2museum.org/war/articles/tsar-bomba-largest-atomic-test-world-history)):

| Effect | Measured | Notes |
|---|---|---|
| Fireball radius | **~3.5 km** | nearly touched the ground from a 4-km airburst; the shock "bounced" it back up |
| Mushroom cloud height | **~67 km** | above the stratosphere, into the mesosphere; cap ~95 km wide |
| Thermal flash (3rd-degree burns) | **~100 km** | could burn skin at 100 km |
| Total structural destruction | **~35 km** | all buildings levelled (~5 psi contour) |
| Window breakage / light damage | **~900 km** | glass broke in Norway and Finland |

## The scaling laws

### Blast overpressure — cube-root scaling

The blast is a strong shock; by **Sedov–Taylor self-similarity** the energy is
released into a *volume*, so the range for a fixed overpressure grows with the
**cube root** of the yield ([Glasstone & Dolan 1977](https://www.atomicarchive.com/resources/documents/effects/glasstone-dolan/glossary.html);
[NukeBlastSimulator methodology](https://nukeblastsimulator.com/methodology)):

```
R(overpressure) = R₀ · (W / 1 kt)^(1/3)      [W in kt]
```

with reference radii (`R₀`, in km per kt^⅓):

| Overpressure | R₀ (km) | Effect |
|---|---|---|
| 20 psi | 0.28 | reinforced concrete destroyed |
| 5 psi | 1.03 | most buildings collapse, **total destruction** |
| 1 psi | 2.93 | window breakage, injuries |

Check: 5 psi at 50 Mt → `1.03 · (5×10⁷)^(1/3) = 1.03 · 368 = 38 km` ≈ the
observed 35-km total-destruction radius. ✓ A 500-Mt Super Tsar multiplies every
blast radius by `10^(1/3) = 2.15×`.

### Fireball radius — the 0.4 power

The luminous fireball's maximum radius scales more steeply, roughly with `W^0.4`
(it is set by the radiating/hydrodynamic balance, not pure blast volume). We
calibrate the constant to the Tsar anchor (3.5 km at 50 Mt):

```
R_fireball = 46 · W^0.4  metres          [W in kt]
```

Check: 50 Mt → `46 · (5×10⁷)^0.4 = 46 · 76 = 3.5 km` ✓. A ×10 yield grows the
fireball by `10^0.4 = 2.5×` → ~8.7 km for the Super Tsar.

### Thermal fluence — third-degree-burn radius

Radiant exposure at a given range falls with distance and atmospheric
transmission; the burn radius likewise scales ≈ `W^0.4`, calibrated to 100 km at
50 Mt:

```
R_thermal = 1316 · W^0.4  metres         [W in kt]
```

→ 250 km for the Super Tsar.

### The shock front in time — Sedov–Taylor

The expanding blast front position is the classic point-explosion similarity
solution ([Sedov–Taylor](https://en.wikipedia.org/wiki/Taylor%E2%80%93von_Neumann%E2%80%93Sedov_blast_wave)):

```
R_shock(t) = ξ · (E t² / ρ)^(1/5),   ξ ≈ 1.03 (air, γ=1.4),  E = W · 4.184×10¹² J
```

This gives the condensation-ring / shock-sphere its animated expansion (very fast
early, then decelerating as `t^(2/5)`).

### The mushroom cloud — buoyant rise

The hot fireball is far less dense than the surrounding air, so it rises by
buoyancy, entraining air into the classic **toroidal cap + stem** and
overshooting to a ceiling `H_max` set by atmospheric stability. We model the rise
as a saturating exponential to the observed ceiling:

```
H(t) = H_max · (1 − e^(−t/τ)),   H_max(50 Mt) ≈ 67 km
```

The Wilson **condensation cloud** (a brief white shell where the negative-pressure
phase of the shock cools humid air below its dew point) rides the shock front.

### Fireball temperature — the two-pulse thermal cooling

The fireball starts at millions of kelvin (soft X-rays), but its *visible*
brightness temperature drops through a hydrodynamic minimum and a second maximum,
then cools ([Glasstone & Dolan](https://www.atomicarchive.com/resources/documents/effects/glasstone-dolan/glossary.html)).
For colour we model an effective blackbody cooling from ~30,000 K (blue-white)
through yellow-white to a dull ~1,500 K red as the fireball ages, feeding
`engine.color.kelvin_to_rgb`.

## Vacuum — a detonation in space (no atmosphere, no mushroom)

With no surrounding air the physics is completely different, and the model
switches laws for `super_tsar_space`:

- **No blast wave.** A shock needs a medium; in vacuum there is nothing to
  compress, so there is no overpressure, no Sedov front, no window-breaking.
- **No incandescent fireball.** The "fireball" of an air burst is *air* heated to
  incandescence; with no air, the energy leaves as a flash of **soft X-rays** and
  the **vaporised bomb debris** expands as a thin plasma shell.
- **No mushroom cloud.** Buoyancy needs gravity *and* a fluid to rise through;
  the debris simply expands roughly spherically and ballistically, thinning and
  cooling. This matches the 1962 **Starfish Prime** high-altitude test (1.4 Mt at
  400 km): no mushroom — an expanding plasma shell, an artificial aurora along the
  geomagnetic field, and a powerful EMP
  ([Wikipedia — Starfish Prime](https://en.wikipedia.org/wiki/Starfish_Prime)).

We model the space burst as a **ballistic debris shell** whose radius grows
`R(t) ≈ v·t` (v set by `√(2E/m_debris)`), radiating as it thins, with the planet
and its gravity in frame but untouched by any blast — the contrast with the
atmospheric burst is the point.

## How it renders

`blast/physics.py` exposes these laws as pure host **and** device functions
(`fireball_radius`, `overpressure_radius`, `thermal_radius`, `shock_radius`,
`mushroom_height`, `fireball_temp`, plus the `BlastParams` for a yield). The
scenes ray-march a **procedural landscape** (heightfield + scattered trees),
composite a **volumetric fireball** (blackbody emission from `fireball_temp` ×
`kelvin_to_rgb`, turbulent fbm density) and a **mushroom cloud** (toroidal cap +
stem), draw the **condensation shock ring** at `shock_radius(t)`, and **flatten +
scorch the trees** inside the damage rings — so the render is not just pretty but
*sized to the physics*. The space scene swaps the mushroom for the ballistic
plasma shell over a planet.

The shock ring's layered look — a crisp bright leading edge trailing into a glow
halo — is the `shock_ring` helper, adopted from the sister project
[`the-virus-block-mc`](https://github.com/cyberpunk042/the-virus-block-mc)'s
`shockwave_ring.glsl` (a ground-following ring: sharp core + inner glow + soft
outer glow).

## Cross-references

- [Research 00 — foundations](00-foundations.md): the raymarch + heightfield the
  landscape reuses.
- Blackbody colour: `engine.color.kelvin_to_rgb` (Research 09 cosmos).
- Primary sources: [Glasstone & Dolan, *The Effects of Nuclear Weapons* (1977)](https://www.atomicarchive.com/resources/documents/effects/glasstone-dolan/glossary.html),
  [Tsar Bomba](https://en.wikipedia.org/wiki/Tsar_Bomba),
  [Sedov–Taylor blast wave](https://en.wikipedia.org/wiki/Taylor%E2%80%93von_Neumann%E2%80%93Sedov_blast_wave),
  [Starfish Prime](https://en.wikipedia.org/wiki/Starfish_Prime).
