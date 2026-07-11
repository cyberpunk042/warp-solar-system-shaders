# Warp Shaders

**A hyper-realistic procedural rendering engine for [NVIDIA Warp](https://github.com/NVIDIA/warp)** (`warp-lang`).

Per-pixel `@wp.kernel` shaders — SDF raymarching, procedural noise, PBR,
physically based atmosphere, and volumetrics — written in Python, JIT-compiled
to **CUDA when a GPU is present and to CPU otherwise**, driven by **one quality
knob** (`--quality low|medium|high|ultra`) so the same scene renders on a laptop
CPU and scales to a high-end GPU. Every technique cites a primary source in-code
and in [`docs/research/`](docs/research/).

```python
import warp as wp
import warp_shaders as ws

wp.init()
ws.set_active("high")                                          # quality tier
img = ws.render("earth_v2", width=1280, height=720, time=0.0)  # (H, W, 3) float

# ...or call the engine directly from your own kernel:
ws.procedural   # noise (value/Perlin/simplex/Worley/fbm/ridged/...) + SDF library
ws.engine       # uniforms, PBR, Material, atmosphere (+LUT), volumetrics, post
ws.textures     # portable 2D/3D/equirect sampling over wp.array
ws.lod          # quality tiers (low/medium/high/ultra)
```

> **📖 Documentation** — the engine has a full manual:
> **[Home](docs/index.md)** ·
> **[Quickstart](docs/quickstart.md)** ·
> **[Concepts](docs/concepts.md)** ·
> **[Writing a scene](docs/guides/writing-a-scene.md)** ·
> **[API reference](docs/api/index.md)** ·
> **[Gallery](docs/gallery.md)**.
> Build the browsable site with `pip install -r docs/requirements.txt && mkdocs serve`.

It's also a **multi-scene gallery**: each shader is one self-contained module in
`warp_shaders/scenes/`, auto-discovered by a registry. Adding a scene is adding
a file — no central list to edit.

## Hyper-realistic engine

A reusable, **tiered, research-grounded rendering engine** — a procedural
toolkit + a render engine with one quality knob so the same scene runs on CPU
here and scales to a high-end GPU. See the [API reference](docs/api/index.md)
for every public symbol.

| noise toolkit | PBR raymarch | atmosphere |
|---|---|---|
| ![noise](docs/engine/noise_gallery.png) | ![pbr](docs/engine/pbr_demo.png) | ![sky](docs/engine/sky.png) |
| **volumetric clouds** | **Earth v2 (flagship)** | **baked-map Earth** |
| ![clouds](docs/engine/clouds.png) | ![earth](docs/engine/earth_v2.png) | ![earth map](docs/engine/earth_map.png) |
| **terrain** | **ocean** | **volumetric nebula** |
| ![terrain](docs/engine/terrain.png) | ![ocean](docs/engine/ocean.png) | ![nebula](docs/engine/nebula.png) |
| **gas giant + rings** | **alien world (twin suns)** | **spiral galaxy** |
| ![gas giant](docs/engine/gas_giant.png) | ![alien](docs/engine/alien.png) | ![galaxy](docs/engine/galaxy.png) |
| **aurora** | **lava planet** | **desert dunes** |
| ![aurora](docs/engine/aurora.png) | ![lava planet](docs/engine/lava_planet.png) | ![dunes](docs/engine/dunes.png) |
| **glacier** | **depth of field** | **slot canyon** |
| ![glacier](docs/engine/glacier.png) | ![depth of field](docs/engine/dof_showcase.png) | ![canyon](docs/engine/canyon.png) |
| **underwater reef** | | |
| ![reef](docs/engine/reef.png) | | |

- **Procedural toolkit** (`warp_shaders/procedural/`) — value/Perlin/Worley/fbm/
  ridged/billow/domain-warp/curl noise **with analytic derivatives**, plus an SDF
  primitive+operator library. Sources: IQ, Gustavson, McGuire, Bridson.
- **Render engine** (`warp_shaders/engine/`) — `@wp.struct` uniforms (camera/light/
  frame/quality), an adaptive sphere-tracing raymarcher, **GGX Cook-Torrance PBR**,
  **physically based atmospheric scattering** (Nishita/O'Neil Rayleigh+Mie **plus
  Hillaire multiple scattering**), a **volumetric cloud** raymarcher (Schneider
  density, Henyey-Greenstein, Beer-Lambert, sun light-march) over a **baked seamless
  3D detail volume**, a **thin-lens depth-of-field** camera, and a host **post**
  pipeline (ACES/AgX tonemap, bloom, **god-rays**, vignette).
- **LOD tiers** (`warp_shaders/lod.py`) — one knob scales raymarch/shadow/AO/atmosphere/
  cloud sample counts, octaves, LUT sizes; auto-detected per device.
- **Textures & LUTs** (`warp_shaders/textures.py`) — portable bilinear sampling over
  `wp.array2d` (CPU+CUDA): equirectangular planet maps (bake once, or drop in a NASA
  **Blue Marble** JPG via `load_equirect`), precomputed **atmosphere transmittance +
  multiple-scattering LUTs**, and baked **3D noise volumes** (`sample3d`) for cheap
  cloud detail and emissive nebulae.

```bash
python render.py --scene earth_v2 --quality high -o earth.png   # the flagship
python render.py --scene sky --quality medium --frames 120 --gif out/day.gif
python render.py --scene pbr_demo --quality ultra -o pbr.png
python -m tests.test_procedural                                  # toolkit tests
```

The earlier scenes and the nuclear/Earth **simulations** now render through the
engine's post pipeline too. Grounded in Warp v1.12+ hardware textures
(`wp.Texture2D/3D`, mipmaps) — precomputed atmosphere LUTs and a Blue Marble map
are the next tier of realism.

The flagship scene is a **neutron star**: a dense pulsar core with relativistic
jets along the magnetic axis, magnetic field rings, orbiting matter, and a
cube-mapped starfield — a Warp port of the GLSL Shadertoy original kept at
[`reference/neutron-star.frag`](reference/neutron-star.frag).

| neutron star | sun | black hole |
|---|---|---|
| ![neutron star](docs/preview.png) | ![sun](docs/sun.png) | ![black hole](docs/black-hole.png) |
| **planet** | **earth** (realistic) | |
| ![planet](docs/planet.png) | ![earth](docs/earth.png) | |

The **earth** scene is a realistic globe: ray-sphere planet with atmospheric
scattering (blue rim + sunset limb), oceans with a specular sun-glint, procedural
continents, drifting clouds, a day/night terminator with night-side city lights,
over a starfield — fully procedural, no texture asset. Shading lives in
[`warp_shaders/earthgfx.py`](warp_shaders/earthgfx.py), shared with the Earth
blast simulation below.

## The atom, from the bottom up

A second, composable strand: build an atom out of its constituents. These scenes
are **physics-informed but stylized**, and each higher level reuses the lower
primitives from [`warp_shaders/particles.py`](warp_shaders/particles.py).

| quark | proton | neutron |
|---|---|---|
| ![quark](docs/quark.png) | ![proton](docs/proton.png) | ![neutron](docs/neutron.png) |
| color charge r→g→b, gluon wisps | up+up+down, color-neutral | up+down+down, color-neutral |

| electron | atom (hydrogen) |
|---|---|
| ![electron](docs/electron.png) | ![atom](docs/atom.png) |
| 1s probability cloud | proton nucleus inside the 1s cloud |

What's modeled (stylized, not to scale):

- **Quark** — a lone quark can't be isolated (confinement), so it's shown as one
  orb whose QCD **color charge** cycles red→green→blue, with gluon wisps.
- **Proton / neutron** — three quarks (`uud` / `udd`) whose red/green/blue color
  charges sum to **color-neutral**, bound by **gluon flux tubes**. Same shared
  `nucleon` primitive; down quarks render dimmer, and the confinement "bag" is
  warm for the proton (+1) vs cool for the neutron (0).
- **Electron** — a point lepton rendered as the hydrogen **1s orbital**
  probability cloud (`exp(-r/a)`), volumetrically integrated with quantum sparkle.
- **Atom** — a proton nucleus wrapped by the electron's 1s cloud. The nucleus is
  exaggerated (a real one is ~1e-5 of the atom) so its structure stays visible.

The build is genuinely bottom-up: `atom` composes the same `nucleon` used by
`proton`, and the same cloud integrator used by `electron`. Heavier atoms (more
nucleons, more electron shells) extend the same primitives.

## Elements — the stylized (non-realistic) aesthetic

A deliberately artistic take on the atom: the iconic neon **Bohr-model** look —
a glowing packed nucleus (warm protons + cool neutrons) wrapped by tilted
electron shells with orbiting electrons. One generic Warp kernel renders **any**
element from runtime parameters (Z protons, N neutrons, electrons-per-shell), so
all elements share one code path.

| H | He | C | O |
|---|---|---|---|
| ![H](docs/elements/hydrogen.png) | ![He](docs/elements/helium.png) | ![C](docs/elements/carbon.png) | ![O](docs/elements/oxygen.png) |
| **Ne** | **Na** | **Cl** | **Ar** |
| ![Ne](docs/elements/neon.png) | ![Na](docs/elements/sodium.png) | ![Cl](docs/elements/chlorine.png) | ![Ar](docs/elements/argon.png) |

**18 elements** (periods 1–3, H → Ar) are registered as scenes, each with the
correct proton/neutron count and shell occupancy (e.g. Ar = 2-8-8):

```bash
python render.py --scene carbon -o carbon.png
python render.py --scene argon  --frames 120 --fps 30 --gif out/argon.gif
python render.py --list          # every element shows up
```

Adding more elements is one row in the data table in
[`warp_shaders/scenes/elements.py`](warp_shaders/scenes/elements.py) — the
kernel already handles any Z / N / shell configuration.

## Simulations — gravity, chain reactions, blasts

Everything above is a *per-pixel shader*. This part uses Warp for what it's
actually built for: **GPU particle simulation**. A stateful particle system
evolves over time under real forces (a Warp kernel integrates gravity, thermal
buoyancy, drag, and cooling each step), driving nuclear and thermonuclear blasts
— **with the full chain reaction**, and an optional **gravity drop** beforehand.

| nuclear (fission) | thermonuclear (fission → fusion) |
|---|---|
| ![nuclear](docs/sim/nuclear.gif) | ![thermonuclear](docs/sim/thermonuclear.gif) |

```bash
python simulate.py --scenario nuclear       --drop --gif out/nuke.gif
python simulate.py --scenario thermonuclear --drop --gif out/thermo.gif
python simulate.py --scenario thermonuclear --no-images   # just the chain-reaction report
```

Each run has three phases:

1. **Drop** — the device falls under gravity (real ballistic motion) to the burst altitude.
2. **Chain reaction** — a fission cascade modelled with self-limiting **point kinetics**: one seed neutron multiplies (`k_eff > 1`) until the fuel burns up and reactivity drops below critical — the characteristic neutron-population *pulse*. `simulate.py` prints the generation-by-generation table:

   ```
   frame |  fission n |  fis.E | fusion n |  fus.E
      29 |      67.49 |  0.000 |     0.00 |  0.000
      39 |   68326.77 |  0.076 |     0.00 |  0.000
      44 |  327225.23 |  0.791 |     0.00 |  0.000   <- fission peaks, fuel burning out
   ```

3. **Fireball** — the released energy spawns a hot particle fireball that expands, then rises by buoyancy against gravity + drag → mushroom cloud (the camera tracks it up).

**Thermonuclear** adds a second stage: once the fission *primary* releases enough energy it **ignites** a fusion *secondary* (the Teller–Ulam idea) — a second, much larger pulse and fireball (~25× the yield here). Physics timescale is dramatised onto frames; energies are arbitrary units for comparison, not megatons.

Runs on CPU here (Warp's CPU codegen); identical on CUDA, in real time. See
[`warp_shaders/sim/`](warp_shaders/sim/) — `engine.py` (particles + integrate
kernel + splat renderer) and `blast.py` (drop + kinetics + fireball).

## Earth — every warhead at once (a sensitization piece)

A gravitationally-bound **particle Earth** under simultaneous global detonation.
Grounded in real numbers, because the truth is sharper than the sci-fi: the
entire arsenal is **~10⁻¹³ of Earth's gravitational binding energy** — the
dinosaur-killer impact was **~26,000× larger** and Earth survived geologically.
So the planet does **not** shatter. What dies is everything living on it.

Three outcomes, chosen one at a time; three arsenals (`current` ~9,500 · `total`
~12,500 re-armed · `peak` ~60,000):

| grounded (real) | toxic (real) | shatter (hypothetical) |
|---|---|---|
| ![grounded](docs/sim/earth_grounded.png) | ![toxic](docs/sim/earth_toxic.png) | ![shatter](docs/sim/earth_shatter.png) |
| planet intact, global flashes | nuclear-winter soot shroud, dead world | alien "softron" energy → rock + ice cloud |

```bash
python simulate_earth.py --arsenal total --outcome grounded  --gif out/earth.gif
python simulate_earth.py --arsenal total --outcome toxic     --gif out/toxic.gif
python simulate_earth.py --arsenal peak  --outcome shatter    --gif out/shatter.gif
python simulate_earth.py --arsenal total --no-images          # the honest report:
```

```
warheads               : 12,500
total yield            : 3,800 Mt  =  1.590e+19 J
arsenal / binding      : 7.09e-14   (need >= 1 to disperse the planet)
blast dv / escape vel  : 2.66e-07   (escape = 11,186 m/s)
dino-killer / arsenal  : 26,416x  (and Earth survived that)
VERDICT: PLANET INTACT. ...
```

- **grounded / toxic** are real physics: the planet is in equilibrium and stays
  put; the arsenal only scorches the surface. They render the **realistic globe**
  (the same `earthgfx` shader as the `earth` scene) with detonation flashes
  clustered over real basing regions; `toxic` greys the surface and layers on the
  firestorm soot shroud (nuclear winter) — the honest catastrophe.
- **shatter** is explicitly labeled non-physical: energy scaled to Earth's
  binding energy (~10¹³× the real arsenal) so the planet disperses; inner debris
  falls back into clumps under self-gravity (`_grav` kernel) — a shambling rock +
  ice cloud. This is the alien-weapon *hypothetical*, not what nukes do.

The point: you never needed to break the planet to end the world on it.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`warp-lang` ships its own CPU/CUDA codegen — no separate CUDA toolkit needed for
CPU rendering. On a machine with an NVIDIA GPU + driver, Warp uses it
automatically.

## Render

```bash
python render.py --list                         # show available scenes

# single frame (auto device: CUDA if available, else CPU)
python render.py --scene neutron_star --time 2.0 --width 1280 --height 720 -o frame.png

# a spinning GIF
python render.py --scene neutron_star --frames 60 --fps 30 --gif out/spin.gif

# PNG frame sequence
python render.py --scene neutron_star --frames 120 --out-dir out/frames

# force CPU (works anywhere; slower)
python render.py --scene neutron_star --device cpu --width 640 --height 360 -o frame.png
```

`--mouse MX MY` drives the camera/pan, matching the shader's `iMouse` convention
(pixel coordinates).

## Add a scene (the workflow for every new shader)

1. Copy the template:

   ```bash
   cp warp_shaders/scenes/_template.py warp_shaders/scenes/my_scene.py
   ```

2. Write the kernel and set the `SCENE` name. Every scene implements the **same
   kernel contract**, which is what keeps the launcher uniform:

   ```python
   @wp.kernel
   def render_kernel(img: wp.array2d(dtype=wp.vec3),
                     width: int, height: int, time: float, mouse: wp.vec2):
       i, j = wp.tid()          # i = row, j = column
       ...
       img[i, j] = wp.vec3(r, g, b)

   SCENE = Scene(name="my_scene", kernel=render_kernel, description="...")
   ```

3. It's live immediately:

   ```bash
   python render.py --list
   python render.py --scene my_scene -o my_scene.png
   ```

Underscore-prefixed modules (like `_template.py`) are skipped by discovery.

### GLSL → Warp cheatsheet

Porting a Shadertoy shader is mostly mechanical. The main friction is that Warp
has no swizzles and distinguishes scalars from vectors:

| GLSL | Warp |
|---|---|
| `mainImage(out vec4 c, in vec2 fragCoord)` | the `render_kernel` body |
| `iResolution` / `iTime` / `iMouse` | `width, height` / `time` / `mouse` kernel args |
| `mix(a, b, t)` | `wp.lerp(a, b, t)` |
| `fract(x)` | `x - wp.floor(x)` (or `sdf.fract`) |
| `atan(y, x)` | `wp.atan2(y, x)` |
| `p.xz = rotate(p.xz, a)` | rebuild: `r = rot2(wp.vec2(p[0], p[2]), a); p = wp.vec3(r[0], p[1], r[1])` |
| `v.x` / `v.y` / `v.z` | `v[0]` / `v[1]` / `v[2]` |
| `void f(out float m)` | return a tuple: `return dist, m` |
| `texture(iChannel0, uv)` (image) | a procedural `@wp.func` (fBm/noise) — see `black_hole.py`'s `nebula_tex` or `sun.py`'s `sun_tex` |
| `texture(iChannel1, ...)` (audio FFT) | dropped — use a fixed constant (no audio) |

**Channel convention.** Shadertoy shaders often read image/audio from
`iChannelN`. This gallery has no bound channels, so ports substitute them:
image textures become procedural noise `@wp.func`s, and audio reactivity is
dropped in favor of a fixed constant (scenes still animate via `time`). That
keeps every scene self-contained and asset-free. (If a scene ever needs a real
image, we can add a texture-array sampling path then — the kernel contract
stays the same.)

Reusable building blocks live in `warp_shaders/sdf.py` (`hash2d`, `noise2d`,
`fbm2d`, `rot2`, `sd_torus`, `fract`). Grow that toolkit as scenes share more
primitives. See `warp_shaders/scenes/neutron_star.py` next to
`reference/neutron-star.frag` for a full worked port.

## Layout

```
render.py                        CLI: per-pixel scenes (--list, --scene, frame / GIF)
simulate.py                      CLI: particle-sim blasts (nuclear / thermonuclear, --drop)
simulate_earth.py                CLI: Earth under global detonation (grounded/toxic/shatter)
warp_shaders/
  scene.py                       Scene contract + auto-discovery registry
  sdf.py                         reusable @wp.func toolkit (hash/noise/fbm/rot/SDF)
  particles.py                   particle primitives (quark/gluon/nucleon + camera + volumetrics)
  earthgfx.py                    realistic-Earth shading (scene + sim share it)
  sim/                           stateful particle simulation (Warp physics)
    engine.py                    ParticleSystem: integrate kernel + splat renderer
    blast.py                     gravity drop + chain-reaction kinetics + fireball
    earth.py                     gravitationally-bound Earth + arsenals + outcomes + report
  scenes/
    neutron_star.py              flagship pulsar scene
    black_hole.py                gravitationally-lensed BH + accretion disk
    planet.py                    lit planet + distant star + lens flare (iq/mu6k)
    sun.py                       trisomie21 star corona (texture -> procedural)
    starfield.py                 minimal scene (registry demo)
    quark.py  proton.py          the atom, bottom-up: quark -> nucleons ...
    neutron.py electron.py atom.py   ... -> electron -> hydrogen atom
    elements.py                  18 stylized Bohr-model elements (one generic kernel)
    earth.py                     realistic Earth from space (uses earthgfx)
    _template.py                 copy-me starter (skipped by discovery)
reference/
  neutron-star.frag              original GLSL shaders (provenance / cross-check)
  black-hole.frag
  planet.frag
  sun.frag
docs/*.png                       rendered stills (one per scene)
requirements.txt
```

## Why Warp instead of a GLSL player

Warp kernels are ordinary Python that JIT-compile to native CUDA/CPU, so the
same raymarcher is scriptable, differentiable-capable, and composes with NumPy
and the rest of a simulation pipeline — while still reading like a shader.
