# Research 45 — Simulation of reality + the virtual compression of visual information

> The graphics card, taken seriously. Two intertwined commitments:
>
> 1. **Simulation of reality — the laws are not facultative.** The card is simulated by *respecting the
>    physics of the world it is made of*: electromagnetism, semiconductor switching, heat. State
>    evolves only by the real governing equations, `t → t+dt`, and whatever emerges is earned from the
>    rules — never hand-placed. Every layer carries a **conservation/consistency test** that proves the
>    step is not lying (energy constant, ∇·field correct, Σcharge conserved, steady-state matches theory).
> 2. **The virtual compression of visual information.** What the card *does* is compress visual
>    information. We build that as three concrete compression algorithms with **hard requirements**,
>    each observable **as a process in time**, each held to information-theoretic law (Shannon entropy,
>    rate–distortion).
>
> The recursion that binds them: **the card compresses visuals into pixels → physics compresses reality
> into law → our simulation compresses the card into computable rules.** One act, three levels, each
> obeying its real law and each verifiable.

---

## Part I — The physical reality of the card (top-down research, cited)

The reference object is a Blackwell-class board (NVIDIA RTX 5090 / RTX PRO 6000, die **GB202**). Seen
honestly it is one object under nested laws. Bottom to top:

### Layer 1 — Electromagnetic substrate (the ground truth)

Everything on the board is a solution of **Maxwell's equations**. In practice it is three coupled
problems on one FR4 substrate:

- **Power delivery.** A 12 V input (12V-2×6 connector, 600 W rating, only ~1.1× per-pin current margin)
  is converted to a **~1 V core rail at ~450 A** by a **~19–24 phase interleaved synchronous buck VRM**
  (DrMOS power stages rated 50–80 A, switching ~0.3–2 MHz). Stepping 12 V→1 V multiplies current ~12×
  (~37 A in → ~450 A out). Efficiency ~85–92% → ~50 W shed as heat in the VRM alone.
- **Signal integrity.** Copper traces are **transmission lines** (single-ended ~50 Ω, PCIe diff-pairs
  85 Ω). PCIe 5.0 = 32 GT/s/lane (16 GHz Nyquist); **GDDR7 uses PAM3** (3 levels, 1.5 bit/symbol) at
  28 Gbps/pin → 512-bit bus → **1,792 GB/s**. Governing reduction of Maxwell: the **telegrapher's
  equations** (∂V/∂z = −(R+jωL)I, ∂I/∂z = −(G+jωC)V), with **skin-effect R ∝ √f** (δ≈66/√f(MHz) µm),
  dielectric loss ∝ f·tanδ, propagation v ≈ 0.47c–0.6c in FR4, reflections Γ=(Z_L−Z₀)/(Z_L+Z₀) at
  every discontinuity.
- **Decoupling / PDN.** A multi-tier capacitor bank (bulk POSCAP → mid MLCC → low-ESL nF caps under the
  BGA) targets **< ~5 mΩ impedance to ~100 MHz** because the VRM can only respond down to its switching
  period; faster di/dt must be supplied by caps physically nearer the die.

Honest solver (from the methods research): **FDTD on a Yee grid** — E and H staggered by half a
cell/half a step, leapfrog update, **Courant limit Sc = cΔt/Δx ≤ 1/√d**; PML/UPML absorbing borders;
or the 1-D **telegrapher leapfrog** for single traces. Verification: **energy Σ(½ε|E|²+½μ|H|²) constant
in a lossless cavity to machine precision**; **∇·(εE)=ρ** preserved by construction; **wave speed = c**
measured from a launched pulse.

### Layer 2 — Semiconductor switching (where computation is born from physics)

GB202: **92.2 billion transistors**, **750 mm²**, TSMC "4N"/N5-family FinFET, **192 SMs** (128 CUDA
cores each, 4 SIMT partitions of 32 lanes), tensor + RT cores, 128 MB L2, 512-bit GDDR7. Boost
~2.4 GHz, ~104.8 TFLOPS FP32, **575 W** (600 W PRO).

The load-bearing law is the **energy tax of switching**: charging a gate capacitance C through any
resistance dissipates **½CV² as heat, and discharging the other ½CV²** — irreducible, thermodynamic,
not an imperfection. Total **P_dyn = α·C·V²·f** plus leakage **P_leak = I_leak·V** (subthreshold current
falls only ~**60 mV/decade @ 300 K**, the Boltzmann floor). This is why lowering V (the V² term) is the
highest-leverage move, and why the DVFS controller trades (V,f) against power and temperature.

The second law: **moving bytes ≫ computing.** An off-chip DRAM access costs ~**1,300–2,600 pJ** vs ~**4 pJ**
for an arithmetic op — **~300–650×** (Horowitz). A model that costs FLOPs but not bytes is dishonest by
two orders of magnitude. Rendering maps to silicon as vertex→raster→pixel→ROP→framebuffer (shaders as
warps on SMs), ray tracing as fixed-function BVH traversal on RT cores, and increasingly as tensor-core
neural reconstruction (DLSS).

### Layer 3 — Heat (the conserved consequence)

**Essentially 100% of electrical power becomes heat in situ** (radiated/acoustic < 0.01%) — so the
heat-source term *is* the power map. Die average **~0.6 W/mm² (60 W/cm²)**, hotspots **1.5–3 W/mm²**;
junction ~90 °C, throttle soft ~83–88 °C, hard ~105–110 °C. Governing law: the **heat equation**
ρc_p ∂T/∂t = ∇·(k∇T) + q''', Fourier flux q=−k∇T, Newton convective BC −k∂T/∂n = h(T−T∞) (forced air
h≈25–250 W/m²·K), radiation negligible (<1%, prove it once via Stefan–Boltzmann). Silicon k falls with
T (~148→80 W/m·K) — a destabilizing positive feedback the solver must capture. Two coupled timescales:
**die ~2 ms**, **cooler ~30 s**.

Honest solver: **diffusion stencil**, explicit FTCS with **Fourier limit αΔt/Δx² ≤ 1/(2d)** (r≤1/6 in
3-D; Δt≈few µs at 50 µm — so go **implicit backward-Euler/Crank–Nicolson** for real-time), non-uniform
die-floorplan source q''', DVFS feedback clamping q''' at the hotspot node. Verification: **steady state
matches the analytic ∇²T=0 solution**; **mode decay rate = αk²**; **discrete first law** (Σρc_p ΔT/Δt =
Σq − boundary flux) closes each step.

### Layer 4 — The card *as a stack of codecs* (its function)

Physically, the card is purpose-built silicon that trades bits against fidelity under one law. Every
stage is a codec (all cited in Research-45 sources):

| Stage | Codec | Ratio | Loss | Access |
|---|---|---|---|---|
| Texture unit | Block compression **BC1–BC7 / ETC2 / ASTC** (4×4 block = endpoints + per-texel indices) | 6:1 (BC1) → **36:1** (ASTC 12×12) | lossy | **fixed-rate → O(1) random access** |
| Memory controller / ROP | **Delta Color Compression** (anchor pixel + deltas per tile) | ~20–40% bandwidth | **lossless** | variable-rate |
| Depth / MSAA | **HTILE Z-plane** (Zmin/Zmax + plane eq) / **FMASK** (sample→plane index) | several:1 / 50–75% | lossless | metadata-driven |
| Display transmitter | **VESA DSC** (DPCM predict + ICH + entropy + rate buffer) | **3:1** | visually lossless (ISO 29170) | near-fixed-rate |
| Tensor cores | **DLSS** (learned temporal reconstruction; render ¼ pixels, up to 4× frames) | 4:1 pixels / up to 4:1 frames | lossy (learned prior) | temporal |
| Rasterizer | **sampling a continuous scene onto a finite grid** | ∞→finite | lossy (aliasing = the distortion) | — |

The single law behind all of it is **Shannon rate–distortion R(D)**: entropy H is the floor for lossless
(DCC/Z/FMASK live there — they remove only redundancy); for D>0 you slide down R(D) (BCn/ASTC/DSC/DLSS).
The fault line is **fixed-rate (O(1) access, suboptimal rate) vs variable-rate (entropy-optimal,
sequential)** — one tradeoff resolved differently under different access constraints. **Rasterization
itself is the original lossy compression**, bounded by **Nyquist–Shannon**, with aliasing as its exact
signature.

**This is the bridge to Part II:** compression is not a metaphor bolted onto the card — it is literally
what the hardware is for. Our three algorithms are new points on the same rate–distortion curve.

---

## Part II — The three compression algorithms (operator spec — hard requirements)

> Operator directive (verbatim, sacrosanct — 2026-07-14):
> *"one compression for example can just merge the same thing together and have digit to represent the
> locations of the vsrious same element that also grow a but the cube part. The other just fold it and
> dont care about the collision, it build the compression image in the process and not at the result, a
> bit like docker image. This one is the folding one, but you have to fold it right and squish it just
> right and you need a really 20x smaller cube of the total surface of the whole item of compression.
> Another is the break down of the item into a web of word that remesent per each atom a word, or a
> token rather in a web that gives values and we can then compress it from DNA equivalent sequence into
> the whole process of chromosome. Those come with real real hard requirement, do not minimize
> anything, we will take as much time and as much round as needed."*

Three algorithms compress **the item** (the card / its visual-information content). Each is observable
**as a process in time** and held to a **hard requirement**.

### C1 — Merge → cube (dedup by identity + location index)

- **Mechanism.** Merge every identical element into a **single stored copy**; store **digits/indices
  recording the locations** of every occurrence of that same element. As the merged core shrinks, the
  **location-index set grows** — that growing index structure *is* "the cube part."
- **Nature.** **Lossless**, computed **at the result** (find identities → dedup → build the location
  table). Rate is governed by entropy: it removes only true redundancy (repeated elements). This is the
  dictionary/dedup point on R(D) at D=0.
- **Prototype in repo.** `warp_compress/wrapfold.py` + `cardfold.py` already merge matching cells and
  keep a "same/different" bitmap; C1 sharpens this to explicit **identity-merge + location-index**.
- **Built and verified (`warp_compress/mergecube.py`, `tests/test_mergecube.py`).** The real board is
  sampled to a 3-D occupancy grid, cut into `block`-sized cubes, and every **identical block is merged
  to one stored copy** (a dictionary of unique pieces); a 3-D grid of **digits** — the location index —
  records which unique piece sits at each block position. Reconstruction places `dictionary[index[p]]`
  everywhere: **exact, lossless** (`decompress(compress(x)) == x` verified on the real board). On the
  board (block 5) 1008 blocks merge to **144 unique pieces** + a **28×3×12 index cube** → **4.8× lossless**;
  on a repetitive card (one tile repeated) 96 blocks merge to **1 unique** → **54×**. The index grid is
  literally "the cube part": a compact cube of digits that **grows** with the number of distinct pieces
  while the merged dictionary stays small. Lossless throughout — the merge is by identity, not by
  collision.

### C2 — Fold → cube (the folding one, **20×**, built in the process)

- **Mechanism.** **Fold it and do not care about collisions** (overlaps are allowed/expected). The
  compressed image is **built during the folding process, not computed at the end — like a Docker image**
  assembled layer by layer as you go. You must **fold it right and squish it just right.**
- **Hard requirement.** The result is a **cube ~20× smaller than the total surface of the whole item.**
  20:1 is the bar. (i.e. surface/extent of the folded cube ≤ 1/20 of the original item's total surface.)
- **Nature.** Process-compression: the fold *is* the encoding; the frame you watch *is* the compression
  step (not a before/after). Collision-agnostic (unlike C1's identity-merge). Prototype: the
  `warp_fold_card` fold-in-half-into-a-cube scene — now bound to the **20× / fold-right / squish-right**
  requirement and the **compress-in-the-process** rule.
- **Built and verified (`warp_compress/foldcube.py`, `tests/test_foldcube.py`).** The real board is
  sampled to a 3-D occupancy grid (from `gpu_board.board_map`), then folded in half on its longest axis
  repeatedly, **merging the two halves on collision** (logical OR — "don't care about the collision").
  That merge is what beats the lossless folding limit (a thin sheet folds only ~2–3× by surface losslessly)
  and squishes the card down to a **cube 20.3× smaller by total surface** (surface = exposed-face count
  Σ6·occ − 2·shared): a 140×12×58 board (surface 22398) folds in **5 collision-agnostic folds** (x,x,z,x,z)
  to a 17×12×14 cube (surface 1102). The result is cube-ish (max/min dim 1.42), the squish concentrates
  the material (occupancy density 0.469 → 0.813), and the compressed image is built **in the process**
  (each fold overlays onto the growing block, Docker-layer style). It is **lossy** at the merges — as the
  spec directs ("don't care about the collision"). Metric used: **total surface** (the operator's words);
  say the word to re-target by volume or byte-count.

### C3 — Tokenize → web → DNA → chromosome

- **Mechanism.** **Break the item down** so that **each atom becomes a word / token**, forming a **web
  (graph) that gives values**. That yields a **DNA-equivalent sequence** of tokens, which then compresses
  **through the whole chromosome process** — wrapping/coiling the strand layer by layer (matching coils
  merge, nested coils condense), exactly the metaphase-chromosome fold.
- **Nature.** Semantic / graph compression → linear sequence → grammar/coil compression. The atoms →
  tokens step is where *meaning* enters (repeated substructures become repeated tokens → repeated
  phrases → merged coils). Prototype: `warp_compress/chromosome.py` (Re-Pair coil) + the strand→chromosome
  fold; C3 adds the **per-atom tokenization into a value-web** as the front end.
- **Built and verified (`warp_compress/tokenchromo.py`, `tests/test_tokenchromo.py`).** The card is
  broken into atoms (blocks) and each unique block becomes a **token** — the C1 dedup *is* the atom→word
  step, and the dictionary of unique pieces *is* the "web that gives values". Reading the tokens in scan
  order is the **DNA-equivalent sequence** (a genome of the card), compressed through the **whole
  chromosome process** — the Re-Pair coil, where repeated token-*phrases* wrap into nucleosome rules layer
  by layer. On the real board (block 4): 1575 atoms over a 202-word vocabulary coil into a chromosome of
  **114 nucleosomes, 14 layers deep** → **5.4× lossless**, and — because the coil catches repeated
  *phrases* (a whole row of identical memory, a repeated VRM motif) that C1's flat index cannot — it
  **beats flat C1 by 1.41×**. Lossless end to end (uncoil → the DNA → place the vocabulary pieces back →
  the exact card). The animated strand→chromosome fold is the `warp_fold_chromo` scene.
- **Future — the semantic lossy tier.** Merge near-synonym tokens (blocks that agree *within tolerance*)
  before coiling — compressing the card's *sense*, not just its exact bytes. The lossy dial noted below.

**Open spec questions (for the operator to steer — flagged, not assumed):**
- What is **"the item"** precisely — the card's 3-D geometry (voxels), its rendered visual output
  (pixels/frames), or its information content (the scene data)? The three algorithms may target different
  representations.
- C1: is the "cube" a literal 3-D cube shape (like C2's) or the abstract dense-core-plus-index structure?
- C2: is the **20×** by surface area, by volume, or by byte-count of the encoding?
- C3: what is an **"atom"** — a component, a voxel, a pixel, a mesh vertex? And what values does the web carry?

---

## Part III — Bottom-up build plan (physics substrate → the three compressions)

Reality is not facultative, so we build **bottom-up from the real substrate** and only reach the
compression apex once the layers beneath it are honest and verified. One verified layer per commit;
every layer inlines its conservation/consistency test; every render is read back and checked.

| # | Layer | Real law | Warp solver | Verification (the "not lying" test) |
|---|---|---|---|---|
| **B1** | EM field substrate | Maxwell | 2-D FDTD, Yee leapfrog, CFL Sc≤1/√d, PML | energy constant in lossless cavity; wave speed = c; ∇·(εE)=ρ |
| **B2** | Signals on traces | Telegrapher | 1-D RLGC leapfrog, skin-R(f) | reflection Γ correct at a Z-step; lossless line conserves energy |
| **B3** | Power / VRM+PDN | Buck + circuit ODEs | symplectic/trapezoidal MNA | V_out=D·Vin; LC energy conserved (R=0); KCL residual→0 |
| **B4** | Charge / current | Continuity ∂ρ/∂t+∇·J=0 | flux-form + Scharfetter–Gummel | Σρ conserved to round-off; Gauss ∮E·dA=Q/ε; positivity |
| **B5** | Heat | Heat equation | implicit diffusion, die-floorplan q''' | steady-state = analytic; mode decay = αk²; first-law balance |
| **B6** | Coupling | multi-rate | global Δt = min(all stability limits); sub-cycle EM under heat | each monitor stays in tolerance across the coupled run |
| **C1** | Merge→cube | entropy (lossless) | identity-merge + location index | exact round-trip; ratio = redundancy removed |
| **C2** | Fold→cube (20×) | process-compression | fold-right + squish-right, collision-agnostic | **cube ≤ 1/20 item surface**; built in the process; reversible |
| **C3** | Tokenize→chromosome | rate–distortion + grammar | per-atom token web → DNA seq → coil | reconstructs item; real ratio; semantic lossy dial bounded |

**Discipline (carried from the operating rules):** research-first with citations; *do not minimize*;
the item is the **real RTX board** (`gpu_board`), never an abstract stand-in; every step visually
verified by reading the PNG; take as many rounds as the hard requirements need. All on branch
`claude/warp-shader-raymarching-3cohzd`, one PR, kept draft for review.

---

## Sources (from the top-down research pass, 2026-07-14)

**Semiconductor / compute:** NVIDIA RTX Blackwell GPU Architecture whitepaper; VideoCardz &
TechPowerUp GB202 die shots (750 mm², 92.2 B); Chips and Cheese Blackwell; Horowitz, *Computing's Energy
Problem* (pJ/op, DRAM vs compute); MIT 6.884 power (½CV², CV²f); IIT-Delhi subthreshold-swing notes
(60 mV/dec). **Electrical / EM:** Wikipedia 12VHPWR; Amphenol Minitek CEM-5 datasheet; TechPowerUp 5090
FE/Suprim/Matrix teardowns; JEDEC GDDR7 (JESD239) + Rambus PAM3; Samtec/Altium PCIe 85 Ω; Wikipedia
Telegrapher's equations; sigcon/EDN skin effect; Altium FR4. **Thermal:** GamersNexus-derived 5090 temps;
Indium/Thermal-Grizzly TIM data; EngineeringToolbox convective h; SCIRP FTCS 2-D (r≤1/4); MIT RSI
stability; ResearchGate/UniversityWafer silicon k(T). **Compression:** Microsoft D3D11 BCn/BC7; Khronos
ASTC (Nystad et al. HPG 2012); AMD GPUOpen DCC + DeepWiki HTILE/FMASK; Hasselgren & Akenine-Möller depth
compression; Rambus/VESA DSC 1.2a; NVIDIA Ada GPU Science + DLSS 3/4; Stanford EE398A & arXiv 1901.07821
rate–distortion. **Methods:** Schneider *Understanding the FDTD Method*; Bérenger PML; Wikipedia FTCS &
von-Neumann stability; Colorado State buck state-space; Scharfetter–Gummel (UT-Dallas/TU-Wien); NVIDIA
Warp docs. Full URLs retained in the session research log.
