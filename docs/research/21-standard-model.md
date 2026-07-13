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

---

# Beyond the first table — mesons, antimatter, exotic atoms, and the hypothetical

The Standard-Model chart above (quarks, leptons, bosons) plus the proton, neutron,
atom and orbitals were the first pass. This section adds the **composite** hadrons
beyond the nucleon, **antimatter**, **charged/exotic atoms**, and the particles
that theory predicts but no experiment has yet seen.

## Mesons — quark + antiquark

A meson is the two-body strong bound state: **one quark and one antiquark**, carrying
colour + anti-colour (e.g. red + anti-red = colour-neutral), held by a single QCD
gluon flux string. Rendered by `subatomic/meson.py: render_meson` (the nucleon's
emission primitives with two centres instead of three).

| Scene | Content | Mass | Note |
|---|---|---|---|
| `pion` | π⁺ = u d̄ | 139.6 MeV | the lightest hadron; Yukawa's carrier of the nuclear force |
| `kaon` | K⁺ = u s̄ | 493.7 MeV | first "strange" particle; long-lived (weak decay) |
| `jpsi` | J/ψ = c c̄ | 3096.9 MeV | charmonium; the 1974 "November Revolution" confirmed charm |
| `upsilon` | Υ = b b̄ | 9460.3 MeV | bottomonium; tightly bound heavy quark pair |

Heavier quarkonia (c c̄, b b̄) sit deeper in the confining potential, so they are
drawn more compact and brighter. Flavour tints follow `field.flavor_color`.

## Antimatter — same mass, opposite charge

Every particle has an antiparticle (Dirac 1928; the positron found by Anderson 1932,
the antiproton at the Bevatron 1955). Rendered by extending the existing lepton and
nucleon renderers with an `anti` flag (`subatomic/annihilation.py` for the event):

| Scene | What | Note |
|---|---|---|
| `positron` | e⁺ | the electron's antiparticle — warm positive-charge field, charge-conjugated (reversed) Coulomb ripples |
| `antiproton` | p̄ = ū ū d̄ | anti-colour antiquarks in a violet confinement bag |
| `annihilation` | e⁻ + e⁺ → γ γ | rest mass → two **back-to-back** 511 keV gamma photons (energy + momentum conservation) |

## Exotic / charged atoms

| Scene | What | Note |
|---|---|---|
| `ion` | a cation | an atom that has lost an electron — depleted electron cloud, the ejected electron streaking away, a net-positive charge halo |
| `positronium` | (e⁻ e⁺) | a hydrogen-like atom of matter + antimatter orbiting their common centre; annihilates in ~0.1 ns (para-Ps) to ~140 ns (ortho-Ps) |

## Hypothetical particles — predicted, never observed

| Scene | What | Status |
|---|---|---|
| `tachyon` | imaginary-mass, faster-than-light particle | drags a **Cherenkov shock cone** of blueshifted light; almost certainly non-physical (violates causality), a useful teaching foil |
| `graviton` | spin-2 quantum of gravity | required if gravity is quantised; ripples a spacetime grid by its **quadrupole** (plus-polarisation) strain — the same strain LIGO measures for classical waves |
| `magnetic_monopole` | isolated magnetic charge | predicted by Dirac (1931, explains charge quantisation) and by GUTs; would give **radial** B-field lines. None found — the field always loops (∇·B = 0 so far) |
| `axion` | ultralight pseudoscalar | proposed by Peccei–Quinn to solve the strong-CP problem; a leading cold-dark-matter candidate. Detectable via the **Primakoff effect** — axion ↔ photon conversion in a magnetic field (ADMX, CAST) |
| `dark_matter` | a WIMP | ~27% of the universe's energy; non-luminous, seen only gravitationally — here through the **lensing** of background starlight into arcs |

## Additional citations

- **C. D. Anderson**, *The Positive Electron*, Phys. Rev. 43 (1933) — the positron.
- **O. Chamberlain, E. Segrè et al.**, *Observation of Antiprotons*, Phys. Rev. 100
  (1955) — the antiproton.
- **J. J. Aubert et al.** & **J.-E. Augustin et al.**, Phys. Rev. Lett. 33 (1974) —
  the J/ψ (charm).
- **P. A. M. Dirac**, *Quantised Singularities in the Electromagnetic Field*, Proc.
  R. Soc. A 133 (1931) — the magnetic monopole + charge quantisation.
- **R. D. Peccei, H. Quinn**, Phys. Rev. Lett. 38 (1977); **F. Wilczek**,
  **S. Weinberg**, PRL 40 (1978) — the axion.
- **G. Feinberg**, *Possibility of Faster-Than-Light Particles*, Phys. Rev. 159
  (1967) — tachyons.
- **Planck Collaboration**, *Cosmological parameters*, A&A 641 (2020) — the dark-matter
  density Ω_c h².

## Baryons beyond the nucleon — the hyperons

Replace the proton/neutron's up and down quarks with **strange** quarks and the
**hyperon** family appears (`subatomic/baryon.py: render_baryon` — the nucleon
field with each quark tinted by flavour as well as colour charge):

| Scene | Content | Mass | Note |
|---|---|---|---|
| `lambda` | Λ⁰ = u d s | 1115.7 MeV | the lightest hyperon |
| `sigma` | Σ⁺ = u u s | 1189.4 MeV | a charged strange baryon |
| `xi` | Ξ⁰ = u s s | 1314.9 MeV | the "cascade", decays in a chain |
| `omega` | Ω⁻ = s s s | 1672.5 MeV | three strange quarks — **predicted** by the quark model's SU(3) decuplet in 1962 and found in 1964, confirming the scheme |
| `delta` | Δ⁺⁺ = u u u | 1232 MeV | a spin-3/2 resonance; three identical up quarks in the same state demanded a new three-valued quantum number — **colour** |

## In the detector — how we actually see particles

| Scene | What | Note |
|---|---|---|
| `bubble_chamber` | curved tracks in superheated liquid | a charged particle boils a trail of bubbles; a magnetic field bends it (radius ∝ momentum, sign ∝ charge). Neutral particles are invisible until they decay into a **V** of charged tracks. (Glaser 1952) |
| `particle_collision` | a collider event display | two beams meet and the energy sprays out as a fan of tracks reconstructed by the detector — the modern successor to the bubble chamber (LHC, etc.) |

## Further citations

- **M. Gell-Mann**, *A Schematic Model of Baryons and Mesons*, Phys. Lett. 8 (1964);
  **G. Zweig**, CERN preprint (1964) — the quark model + the Ω⁻ prediction.
- **V. E. Barnes et al.**, *Observation of a Hyperon with Strangeness Minus Three*,
  Phys. Rev. Lett. 12 (1964) — the Ω⁻.
- **O. W. Greenberg**, Phys. Rev. Lett. 13 (1964) — colour, from the Δ⁺⁺ paradox.
- **D. A. Glaser**, *Some Effects of Ionizing Radiation on the Formation of Bubbles
  in Liquids*, Phys. Rev. 87 (1952) — the bubble chamber.
