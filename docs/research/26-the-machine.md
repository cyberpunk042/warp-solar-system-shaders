# Research 26 — The machine: technology & computation

> From the electron and the crystal to the computer. How a machine thinks,
> bottom-up: the switch, the gate, the processor, memory, the network, and the two
> frontiers — quantum and AI.

## The transistor — the switch

A **MOSFET** is a voltage-controlled switch: a **gate** over a silicon channel
between **source** and **drain**. Raise the gate voltage and a conducting channel
forms (ON, "1"); drop it and the channel vanishes (OFF, "0"). Billions of these,
each ~nanometres, are etched into a chip. Everything else is switches wired together.

## Logic gates

Switches combine into **gates** that compute boolean functions:

| Gate | Output is 1 when… |
|---|---|
| NOT | input is 0 |
| AND | both inputs are 1 |
| OR | either input is 1 |
| XOR | inputs differ |
| NAND | *not* (both 1) — **universal** (any circuit from NANDs alone) |

Gates chain into adders, multiplexers, latches — arithmetic and memory.

## The CPU

A **processor** is a city of blocks on a die: the **ALU** (arithmetic/logic), the
**register file**, the **control unit** and **cache**, ticked by a **clock** at
billions of cycles/second. It runs the fetch → decode → execute → write-back cycle,
data flowing along **buses** like traffic on a grid — which is why a chip die,
seen from above, looks like an aerial city.

## Binary & memory

All of it is **binary** — patterns of 0s and 1s. Eight bits make a **byte**; memory
is a vast grid of cells, each holding a bit, addressed by row and column. Data
**flows** as pulses along wires: the machine is patterns of electricity moving
through the switch-lattice.

## The internet

Computers talk by chopping messages into **packets** that hop, router to router,
across a mesh network, each packet independently routed to its destination and
reassembled — a global graph with data streaming along its edges (TCP/IP).

## The frontiers

- **Quantum computer** — a **qubit** is not 0 *or* 1 but a **superposition** of
  both, a point on the **Bloch sphere**; entangled qubits explore many states at
  once. We render the Bloch sphere with a precessing state vector.
- **AI / neural training** — a network of weighted connections whose weights are
  tuned by **gradient descent** to minimise error: a loss landscape a ball rolls
  down, the network lighting up as it learns.

## Rendering approach

| Scene | Technique |
|---|---|
| **transistor** | an SDF MOSFET (gate/source/drain/channel) with the channel glowing when switched ON |
| **logic_gates** | glowing gate-body SDFs with input/output wires carrying 0/1 pulses |
| **cpu_die** | a chip die as an aerial "city" of functional blocks with data pulses on the buses (buildings-style domain repetition) |
| **data_flow** | a memory grid / bitstream — cells flipping, pulses streaming down the wires |
| **internet** | a router graph with packets hopping along the edges (neural-net-style graph) |
| **quantum_computer** | a Bloch sphere with a precessing qubit state vector + entangled partner |
| **ai_training** | a loss-landscape surface with a descending ball + a lighting-up network inset |

Reuses `procedural.noise`, `engine.intersect`, `subatomic.field.sd_capsule`, the
buildings domain-repetition pattern, and `engine.post`.

## Citations

- J. Bardeen, W. Brattain, W. Shockley (1947–48) — the transistor.
- C. Mead & L. Conway, *Introduction to VLSI Systems* (1980) — chip design.
- C. Petzold, *Code: The Hidden Language of Computer Hardware and Software* (2000).
- V. Cerf & R. Kahn, *A protocol for packet network intercommunication*, IEEE Trans.
  Comm. (1974) — TCP/IP.
- M. Nielsen & I. Chuang, *Quantum Computation and Quantum Information* (2000) —
  qubits, the Bloch sphere.
