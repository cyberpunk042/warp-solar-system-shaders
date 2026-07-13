# Research 38 — Electricity in motion (current that flows and does work)

> The [electronics rounds](36-boards-and-memory-blocks.md) built the hardware and then
> pushed electrons through a GPU until it blew up. This round is **electricity itself** —
> charge in motion, the fields it makes, and the work it does: current pulsing down a wire,
> a capacitor charging, a motor turning, a transformer coupling, an arc jumping a gap, and
> the grand discharge — **lightning**. The first massive PR of a four-part arc
> (electricity → engine leap → physics sims → the living body).

This is a **field / emission** round: conductors and machines are the stage, and the show is
glowing charge — accumulated as emission along the camera ray. A toolkit,
`warp_shaders/electric.py`, holds the shared pieces.

## The physics being drawn

### Current — drift vs. signal

A wire carries **current** *I* = *dQ/dt*. The individual electrons *drift* astonishingly
slowly (~mm/s in copper), but the **signal** and the **energy** race down the wire at a
large fraction of *c* — because it is the electromagnetic **field** around the conductor
(the Poynting flux) that actually carries the power, not the bulk motion of the charges.
So the sim shows two things travelling: slow-ish glowing **pulses** of charge, and the fast
bright **flash** of the field switching on. `electric.current_pulse` is the travelling
pulse; a steady thread plus bright moving crests, fading up with the drawn `level`.

### Ohm and Kirchhoff

A resistor obeys **V = IR**; around any loop the voltages sum to zero (Kirchhoff) and at any
node the currents balance. These are the rules a real circuit solver would enforce; here they
set the *look* — brighter where more current flows, hotter where more power *I²R* is
dissipated (the blackbody `heat_color` ramp from the singularity round reused for hot wires
and arcs).

### Capacitance and inductance — storing the field

- A **capacitor** stores energy in the **electric field** between its plates: *Q = CV*,
  *E = ½CV²*. Charging follows the RC curve *V(t) = V₀(1 − e^(−t/RC))* — current rushes in,
  then tapers as the field builds and opposes it.
- An **inductor** (a coil) stores energy in the **magnetic field**: *E = ½LI²*, and resists
  *changes* in current (*V = L dI/dt*). Together, L and C **oscillate** (an LC tank rings at
  *f = 1/2π√(LC)*) — the beating heart of the Tesla coil and every radio.

### Induction — the transformer and the motor

Faraday's law: a **changing** magnetic flux induces a voltage, *EMF = −dΦ/dt*. That single
law runs the grid:

- A **transformer** couples two coils through an iron core; AC in the primary makes a
  changing flux that induces AC in the secondary, stepping voltage up or down by the turns
  ratio.
- A **motor** turns current into torque: current in a coil sitting in a magnetic field feels
  a force (*F = IL×B*); a commutator flips the current each half-turn so the torque keeps
  pushing the same way, and the rotor spins.

### Dielectric breakdown — the arc and the bolt

Push the field past a material's breakdown strength (~3 MV/m for air) and it **ionises** into
a conducting plasma channel — an **arc**. The channel doesn't go straight: it gropes forward
in **steps**, branching, following the path of least resistance through the randomly varying
air (a **stepped leader**). When it bridges the gap the **return stroke** dumps the charge in
a blinding flash, and the channel flickers as successive strokes re-light it. `electric.generate_bolt`
models exactly this — recursive **midpoint displacement** with **branching**, the standard
fractal model of lightning — and the `lightning` scene flashes strike after strike across a
storm sky. The same mechanism, small, is the spark across a gap, the plasma globe's filaments,
and the streamers off a Tesla coil's toroid.

## The toolkit — `electric.py`

- `pt_glow(ro, rd, p, width)` / `seg_glow(...)` — a glowing point / segment seen along a ray
  (Gaussian in the ray's closest approach). Dense points make a glowing filament — the whole
  basis for wires, arcs, and bolts.
- `current_pulse(u, time, speed, level)` — travelling current pulses along a path fraction.
- `corona(ro, rd, c, r)` — the soft glow around a charged conductor or an arc terminus.
- `generate_bolt(a, b, seed, …)` — host-side fractal lightning (midpoint displacement +
  branching), returned as dense points; deterministic per `seed` so a strike is stable within
  its flash and re-rolls on the next.

## Scenes

`lightning` (the fractal cloud-to-ground bolt, strike after strike) — the round's opener.
*(more to come as the strand grows: capacitor charge/discharge, an electric motor, a
transformer, a Tesla coil, a plasma globe, an arc across a spark gap, the power grid.)*

## Sources

- Drift velocity vs. signal/energy propagation, Poynting flux in a conductor — standard
  electromagnetism (Griffiths, *Introduction to Electrodynamics*).
- Ohm's law, Kirchhoff's laws, RC/RL/LC transients — standard circuit theory (Sedra & Smith;
  Horowitz & Hill, *The Art of Electronics*).
- Faraday induction, the DC motor and transformer — standard electromagnetism / machines.
- Fractal lightning as recursive midpoint-displacement with branching (stepped-leader
  dielectric breakdown) — the well-known procedural-lightning model; air breakdown ~3 MV/m.
