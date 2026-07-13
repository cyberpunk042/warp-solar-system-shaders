# Research 28 — The deep ocean

> Descend past the sunlit shallows into the midnight zone: crushing pressure, near-
> total darkness, and life that makes its own light. The largest habitat on Earth,
> and the least seen.

## The zones

Sunlight fades fast in water. Below ~200 m is the **twilight (mesopelagic)** zone,
below ~1000 m the **midnight (bathypelagic)** zone — no sunlight at all, ~4 °C,
hundreds of atmospheres of pressure. The seafloor (abyssal plain) sits near 4000 m;
the deepest trenches (the **Mariana Trench**, Challenger Deep) reach ~11000 m — deeper
than Everest is tall. Down here the only light is **biological**.

## Bioluminescence

Most deep-sea animals make light: the enzyme **luciferase** oxidises a substrate
(**luciferin**), releasing a cold blue-green photon (~470–490 nm — the wavelength that
travels furthest in seawater). Uses: lures (the **anglerfish**'s glowing esca),
counter-illumination camouflage, startle flashes, and mate signalling. Blue-green
dominates because red light is absorbed within metres — so most animals can't even
*see* red, and a few predators that emit red hunt with a private searchlight.

## Hydrothermal vents — life without the sun

At mid-ocean ridges, seawater seeps into the crust, is superheated by magma, and
erupts back as mineral-laden plumes. **Black smokers** billow iron-sulphide particles
at ~350 °C. Around them thrives an ecosystem powered not by sunlight but by
**chemosynthesis**: bacteria oxidise hydrogen sulphide, feeding giant **tube worms**
(*Riftia*), clams and shrimp — an entire food web independent of the sun (discovered
at the Galápagos Rift, 1977).

## Creatures of the deep

- **Jellyfish / siphonophores** — pulsing translucent bells, some (siphonophores like
  *Praya*) forming glowing colonial chains tens of metres long.
- **Anglerfish** — a lure dangling a bioluminescent bulb before a cavernous mouth.
- **Comb jellies (ctenophores)** — rows of beating cilia that refract light into
  running rainbows.
- **Whale fall** — a dead whale sinking to the abyss feeds a succession of scavengers,
  bone-eating worms and bacterial mats for decades: an oasis in the desert.

## Coral reefs — the shallow city

By contrast the sunlit **coral reef** is the ocean's most crowded city: colonial
polyps building calcium-carbonate skeletons in branching, brain, and fan forms, in
symbiosis with photosynthetic **zooxanthellae**, sheltering a quarter of all marine
species.

## Rendering approach

| Scene | Technique |
|---|---|
| **hydrothermal_vent** | a black-smoker chimney SDF + a hot volumetric plume (fBm advection) + tube-worm capsules, in near-black water |
| **bioluminescent** | a dark volume seeded with drifting glowing plankton/creatures (point emitters + curl drift) |
| **jellyfish** | a translucent pulsing bell (SDF, rim-lit) with trailing glowing tentacles, orbiting |
| **coral_reef** | a raymarched reef bed of branching/brain-coral SDFs, colourful, with light shafts + fish motes |
| **mariana_trench** | a descent: layered dark water with falling marine snow and a distant vent glow, steep trench walls |
| **whale_fall** | a whale skeleton on the abyssal floor haloed by scavenger glow + bacterial-mat bloom |

Reuses `procedural.noise` (fbm3, curl3, worley), `subatomic.field` (sd_capsule, void),
`engine.intersect`, `engine.post`, and the emission-absorption volumetric pattern.

## Citations

- W. Beebe, *Half Mile Down* (1934) — first human descent into the deep.
- E. Widder, *Bioluminescence in the Ocean* — luciferin/luciferase, blue-green emission.
- J. Corliss et al., *Submarine Thermal Springs on the Galápagos Rift*, Science (1979)
  — hydrothermal vents & chemosynthesis.
- C. Smith & A. Baco, *Ecology of whale falls at the deep-sea floor* (2003).
- J. Piccard & D. Walsh (1960) — Challenger Deep descent (Trieste).
