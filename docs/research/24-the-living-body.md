# Research 24 — The living body

> A high-realism pass on the life strand at the scale of cells, tissues and
> organs: the neuron and the network that is the mind, the beating heart, DNA
> being read, and the cells of the blood.

## The neuron & the action potential

A neuron has a cell body (**soma**), many branching **dendrites** (inputs), and one
long **axon** (output). It signals with an **action potential** — a self-propagating
electrical spike: the membrane voltage flips from −70 mV to +40 mV and back
(Na⁺ in, then K⁺ out) and the pulse races down the axon at up to ~100 m/s, jumping
between the myelin gaps (nodes of Ranvier). At the far end, **synapses** release
neurotransmitters onto the next neuron. We render the spike as a **bright pulse
travelling down the axon**.

## The network — the mind

~86 billion neurons, each wired to thousands of others, fire in cascading patterns:
that traffic of signals **is** thought. We render it as a graph of nodes with
**pulses propagating along the edges** — when enough inputs arrive together a node
**fires**, lighting its outgoing edges (integrate-and-fire). The living, shifting
pattern is the metaphor for the mind.

## The beating heart

The heart is a muscular pump: it **contracts** (systole) to eject blood and
**relaxes** (diastole) to refill, ~1 Hz at rest, driven by a wave of electrical
excitation from the sinoatrial node. We render a heart-shaped body that **pulses**
in size and glows with each beat (the classic *lub-dub*).

## DNA transcription

To use a gene, the cell **transcribes** it: an enzyme (**RNA polymerase**) clamps
onto the DNA **double helix**, unzips a stretch, reads one strand, and threads out a
complementary **messenger-RNA** copy — which ribosomes later translate into protein.
We render the twin phosphate backbones + base-pair rungs of the helix, with the
polymerase travelling along and an mRNA strand emerging.

## Cells of the blood

- **Red blood cells** (erythrocytes) — biconcave discs packed with haemoglobin,
  carrying O₂; they tumble through vessels in their millions.
- **White blood cells** (leukocytes) — the immune system: a macrophage chases and
  engulfs a bacterium (chemotaxis + phagocytosis).

## Rendering approach

| Scene | Technique |
|---|---|
| **neuron** | soma sphere + branching dendrite/axon capsules (SDF), an emissive pulse travelling down the axon |
| **neural_net** | a node-and-edge graph with pulses propagating along edges; nodes fire (integrate-and-fire) and flash |
| **heartbeat** | a heart-shaped SDF (two spheres + a cone/blob) pulsing in scale + emission on a ~1 Hz beat |
| **dna_transcription** | a ray-marched double helix (two backbones + rungs) with a polymerase bead travelling + an mRNA strand emerging |
| **red_blood_cells** | biconcave-disc SDFs tumbling through a plasma-lit vessel |

Reuses `subatomic.field.sd_capsule`, the SDF sphere-trace pattern, `procedural.noise`,
and `engine.post`.

## Citations

- A. L. Hodgkin & A. F. Huxley, *A quantitative description of membrane current…*,
  J. Physiol. 117 (1952) — the action potential.
- E. R. Kandel et al., *Principles of Neural Science*, 5th ed. — neurons, synapses,
  networks.
- F. Crick, *Central dogma of molecular biology*, Nature 227 (1970) — DNA → RNA →
  protein.
- Guyton & Hall, *Textbook of Medical Physiology* — the cardiac cycle, blood cells.
