# Research 21 — The Standard Model, high quality & realism

> Grounding for the sub-atomic strand: what the fundamental particles *are*, the
> physics that gives each its look, and how we render them as volumetric fields
> rather than the earlier glow-blobs. Operator directive: *"return to the
> sub-atomic particle, I wanna see everything, and in high quality and realism."*

## The Standard Model in one table

Seventeen fundamental particles: **6 quarks**, **6 leptons**, **4 gauge bosons**,
and the **Higgs**. Masses from the Particle Data Group (PDG 2024); charges in units
of the elementary charge *e*; all fermions are spin-½, gauge bosons spin-1, Higgs
spin-0.

### Quarks (spin ½, colour-charged, fractional electric charge)

| Gen | Flavour | Symbol | Charge | Mass (approx) |
|---|---|---|---|---|
| I | up | u | +2/3 | 2.2 MeV |
| I | down | d | −1/3 | 4.7 MeV |
| II | charm | c | +2/3 | 1.27 GeV |
| II | strange | s | −1/3 | 93 MeV |
| III | top | t | +2/3 | 173 GeV |
| III | bottom | b | −1/3 | 4.18 GeV |

Quarks carry **colour charge** (red / green / blue) and are never seen free
(**confinement**); they bind into colour-neutral **hadrons**.

### Leptons (spin ½, colourless)

| Gen | Lepton | Symbol | Charge | Mass |
|---|---|---|---|---|
| I | electron | e⁻ | −1 | 0.511 MeV |
| I | electron neutrino | νₑ | 0 | < 1 eV |
| II | muon | μ⁻ | −1 | 105.7 MeV |
| II | muon neutrino | ν_μ | 0 | < 1 eV |
| III | tau | τ⁻ | −1 | 1777 MeV |
| III | tau neutrino | ν_τ | 0 | < 1 eV |

Charged leptons feel electromagnetism; neutrinos feel only the weak force, are
nearly massless, and **oscillate** between flavours as they travel.

### Bosons (force carriers + Higgs)

| Boson | Symbol | Charge | Mass | Force / role |
|---|---|---|---|---|
| photon | γ | 0 | 0 | electromagnetism (infinite range) |
| gluon | g | 0 | 0 | strong force; carries colour+anticolour (8 states) |
| W | W± | ±1 | 80.4 GeV | weak force (charged current); mediates β-decay |
| Z | Z⁰ | 0 | 91.2 GeV | weak force (neutral current) |
| Higgs | H | 0 | 125 GeV | the Higgs field's excitation; gives mass |

## The physics that gives each its look

- **Colour & confinement (QCD).** The strong force between quarks does **not**
  weaken with distance — the gluon field collapses into a narrow **flux tube**
  (a "colour string") of roughly constant energy per length (string tension
  κ ≈ 1 GeV/fm ≈ 16 tonnes of force). Pull two quarks apart and the tube stores
  energy until it snaps into a new quark–antiquark pair. Lattice-QCD simulations
  show this tube directly. → we render the gluon field as a **taut, textured,
  flowing tube** between quarks, not a fuzzy line.
- **Asymptotic freedom.** Up close the strong coupling is *weak*, so quarks
  rattle quasi-freely inside the hadron "bag" (MIT bag model) → quarks jitter
  inside a confinement bag.
- **Proton = uud, neutron = udd.** Three **valence** quarks in a sea of gluons and
  virtual quark–antiquark pairs. The proton is +1, the neutron 0 (2/3+2/3−1/3 = 1;
  2/3−1/3−1/3 = 0). Their size is ~0.84 fm (proton charge radius).
- **Hydrogen orbitals.** An electron bound to a proton is a standing probability
  wave |ψ_{nlm}|². The wavefunction factorises into a radial part R_{nl}(r)
  (associated Laguerre polynomials, scale = Bohr radius a₀ = 0.529 Å) and an
  angular part Y_{lm}(θ,φ) (spherical harmonics):
  - **1s** (n=1,l=0): spherical, density ∝ e^{−2r/a₀}.
  - **2p** (n=2,l=1): two lobes along an axis (dumbbell), a node at the nucleus.
  - **3d** (n=3,l=2): four-lobed cloverleaf (and the dz² torus+lobes).
  → we render the true |ψ|² as a **volumetric density**, integrated along the ray,
  so the orbital's real shape (nodes, lobes) appears.
- **Electroweak / massive bosons.** The photon and gluon are massless (long tubes /
  waves); the W and Z are heavy and **decay almost instantly** (lifetime ~3×10⁻²⁵ s)
  → render them as a dense, short-lived packet throwing decay products.
- **Higgs.** An excitation of the all-pervading Higgs field → a central boson over a
  faint field lattice, decaying to pairs (bb̄, γγ, ZZ*).

## Rendering approach — from glow-blobs to fields

The earlier scenes drew each particle as one or a few `emitter()` gaussians. The
upgrade renders **physical fields volumetrically**:

| Ingredient | Technique |
|---|---|
| quark | a small emissive core + a turbulent **colour-charge plasma** shell (fBm-modulated), size ∝ log(mass) |
| gluon flux tube | a capsule SDF between two quarks, surface textured by flowing 1-D noise, colour = quark colour blend, pulsing along its length |
| confinement bag | a soft translucent shell (rim-lit) containing the quarks — the hadron boundary |
| hydrogen orbital | ray-marched **volume emission** of the analytic |ψ_{nlm}(r,θ,φ)|², self-coloured by density; nodes render as gaps |
| lepton | tight bright core + a dipole EM field haze; neutrino = a near-invisible shimmer |
| photon | a travelling **transverse E/B wave packet** (oscillating ribbon) |
| W/Z/Higgs | a massive core + decay-product jets emitted on a timer |
| post | additive HDR accumulation → `engine.post.bloom` + ACES tonemap; a dark, faintly star-dusted void so the particle reads as isolated |

All of it reuses the engine's `post`, `intersect`, `color`, `procedural.noise`
and the particle camera helpers; the new physics lives in a
`warp_shaders/subatomic/` package so the old `particles.py` primitives stay
intact for the elements' Bohr atoms.

## Reuse table

| Need | Reuse |
|---|---|
| ray/sphere, capsule | `engine.intersect` + a new `sd_capsule` |
| blackbody / ramps | `engine.color` |
| fBm / ridged / value noise | `procedural.noise` |
| bloom + ACES tonemap | `engine.post` |
| camera orbit | `particles.orbit_ro` / `camera_ray`, or `engine.uniforms` |
| starfield void | a dimmed `earthgfx.stars` or a local sparse-star fn |

## Citations

- **Particle Data Group**, *Review of Particle Physics*, Prog. Theor. Exp. Phys.
  2024 — particle masses, charges, widths.
- D. Griffiths, *Introduction to Elementary Particles*, 2nd ed. — quark content,
  QCD colour, confinement, electroweak bosons.
- D. Griffiths, *Introduction to Quantum Mechanics*, 3rd ed., ch. 4 — hydrogen
  wavefunctions ψ_{nlm}, radial R_{nl}, spherical harmonics Y_{lm}.
- G. S. Bali, *QCD forces and heavy quark bound states*, Phys. Rept. 343 (2001) —
  lattice-QCD flux tube + string tension κ ≈ 1 GeV/fm.
- A. Chodos et al., *New extended model of hadrons* (MIT bag model), Phys. Rev. D9
  (1974) — the confinement "bag".
