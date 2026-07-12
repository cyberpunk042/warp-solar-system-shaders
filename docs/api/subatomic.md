# `warp_shaders.subatomic` — Standard-Model particle fields

Physically-grounded volumetric renderers for the fundamental particles. The field
primitives live in `subatomic.field`; per-particle renderers compose them.
Background: [Research 21 — The Standard Model](../research/21-standard-model.md).

## Field primitives — `subatomic.field`

| Symbol | Kind | Purpose |
|---|---|---|
| `sd_capsule(p, a, b, r)` | device | signed distance to a capsule (gluon tubes) |
| `quark_emit(p, c, r, col, t, seed)` | device | a colour-charged quark **plasma** (hot core + turbulent fBm/ridged shell) |
| `tube_emit(p, a, b, r, col, t)` | device | a QCD **gluon flux tube** — a taut, flowing colour string |
| `bag_glow(p, c, r, tint)` | device | the confinement **bag** shell (MIT bag model) |
| `orbital_psi2(p, orb, a0)` | device | analytic hydrogen density \|ψ_{nlm}\|² (1s/2s/2p/3p/3d z²/3d cloverleaf) |
| `flavor_color(flav)` / `color_charge(k)` | device | six flavour tints / QCD red-green-blue |
| `void(rd)` | device | a near-black, star-dusted background |

## Renderers

| Module | Host functions | Scenes |
|---|---|---|
| `subatomic.hadron` | `render_nucleon(...is_proton)` | `proton`, `neutron` |
| `subatomic.atom` | `render_orbital`, `render_named(orb)` | `atom`, `orbitals` |
| `subatomic.quark` | `render_quark(flav)` | `quark`, `quark_up`…`quark_bottom` |
| `subatomic.lepton` | `render_lepton(kind)` | `electron`, `muon`, `tau`, `neutrino_e/mu/tau` |
| `subatomic.boson` | `render_photon/gluon/w/z/higgs` | `photon`, `gluon`, `w_boson`, `z_boson`, `higgs` |
| `subatomic.render` | `orbit_camera`, `finish` | (shared camera + HDR finish) |

Plus the composite scenes `standard_model` (the full chart) and `beta_decay`
(n→p+e⁻+ν̄ₑ, animated) under `warp_shaders/scenes/`.

Everything is stylised but structurally faithful: correct quark content, colour
summing to neutral, real orbital nodes + lobes, mass-scaled sizes. Not to physical
scale (a nucleus is ~1e-5 of an atom).
