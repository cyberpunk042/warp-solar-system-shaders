# Holographic quantum information — threads, networks, and the code

Research 46 ended on the claim that **entanglement builds geometry**. This set is the
HOW — three mechanisms, three scenes, each backed by an engine computation you can run:
`holo_bit_threads` (entropy as a maximum flow, with an actual max-flow solver),
`holo_mera` (the tiling as a renormalization network, with the log law counted in
beads), and `holo_code` (spacetime as a quantum error-correcting code, with an actual
`[[5,1,3]]` stabilizer code verified over GF(2)). Engine core:
`warp_shaders/engine/holoinfo.py`.

## Part I — entropy is a flow: `holo_bit_threads`

Ryu–Takayanagi computes S(A) as a minimal *cut*. Freedman and Headrick
([1604.00354](https://arxiv.org/abs/1604.00354)) dualized it: S(A) is equally the
**maximum flux** of a divergence-free vector field of "bit threads" with cross-section
≤ 1/4G leaving the region — and the equivalence is nothing deeper (or shallower) than
**max-flow = min-cut**, the LP-duality theorem of network theory. Each thread is one
Bell pair's worth of correlation between A and its complement; the RT surface is just
the bottleneck where the bundle saturates.

The engine doesn't take duality on faith — it runs it (`build_disk_graph`,
`interval_max_flow`, a hand-rolled Dinic): a polar grid on the Poincaré disk whose edge
capacities are the **hyperbolic lengths of the crossed dual segments**, so that a graph
cut measures hyperbolic length and the min cut converges to the RT geodesic. The suite
asserts, per interval size: flow == cut to 1e-9 (exact MFMC), flow monotone in the
interval, and flow tracking the analytic geodesic length `2·ln(sin(Δθ/2)) + const`
(constant = the grid's UV offset) to within the discretization.

The scene draws the dual picture: nested, non-crossing cyan threads pairing points of A
with the complement — their count proportional to S(A), switching on as the interval
grows — squeezing through the white-violet RT bottleneck.

## Part II — the tiling was a tensor network all along: `holo_mera`

Swingle's 2009 observation ([0905.1317](https://arxiv.org/abs/0905.1317)): the MERA
tensor network that renormalizes a critical spin chain — alternating disentanglers and
coarse-grainers, one layer per RG step — has the connectivity of a discretized
hyperbolic plane. Radial direction = RG scale. The `{7,3}` tiling this whole set draws
IS a MERA: UV lattice at the rim, deep IR at the centre.

The dictionary entry is a *counting* fact (`mera_layers`, `mera_cut_bonds`,
test-asserted): an interval of ℓ sites closes its causal cone in `ceil(log₂ ℓ)` layers,
and the minimal cut through the network severs `2·ceil(log₂ ℓ) + O(1)` bonds — doubling
the interval adds exactly one severed bond per side. Each bond carries at most ln χ of
entanglement, so `S ≲ 2·log₂ ℓ · ln χ`: **the CFT log law, produced by counting**. RT is
the continuum limit of the minimal cut.

The scene: tiling generations banded UV-gold → IR-blue, the interval's causal cone
tinted violet to its closing depth, and the minimal cut drawn as **beads at equal
hyperbolic spacing along the RT geodesic — one bead per severed bond**, their number
growing logarithmically as the interval sweeps (and cutoff-dependent exactly as S is:
the beads stop at the same UV cutoff that regularizes the entropy).

## Part III — the bulk is a code: `holo_code`

Why can a bulk operator deep in AdS be represented on *many different* boundary regions,
none of them individually essential? Almheiri, Dong and Harlow
([1411.7041](https://arxiv.org/abs/1411.7041)): because bulk reconstruction has the
structure of **quantum error correction** — the bulk is the logical subspace, the
boundary is the physical qubits, and erasing part of the boundary is a correctable
erasure as long as the bulk point stays inside the intact region's entanglement wedge.
Pastawski–Yoshida–Harlow–Preskill ([1503.06237](https://arxiv.org/abs/1503.06237))
built the toy model: pentagons `{5,4}`, one **[[5,1,3]] perfect-code tensor** per tile
(Laflamme–Miquel–Paz–Zurek,
[quant-ph/9602019](https://arxiv.org/abs/quant-ph/9602019)), contracted into a
holographic code.

The engine carries the actual code: `five_qubit_stabilizers` (cyclic `XZZXI` over
GF(2)) and `erasure_correctable` — the exact criterion (an erasure is correctable iff no
logical operator is supported on it), brute-forced over all Paulis on the erased set.
Asserted: **any 2 erasures correctable, any 3 fatal** — the quantum-MDS / no-cloning
boundary. And the cross-check that makes the scene honest: the geometric wedge rule
(`happy_central_recoverable`: the central bulk qubit survives iff ≥3 of its 5 legs reach
intact boundary) is asserted **equal to the algebraic criterion on the same erased
sets** — the wedge rule and the code rule are the same fact.

The scene erases the boundary and watches: the `{5,4}` pentagon tiling (its own fold,
`p=5, q=4`, same reflection-group construction as the `{7,3}`) glows gold; a crimson
erasure arc sweeps the rim; the intact wedge shrinks; the central tensor's legs die one
by one; and the central logical qubit goes dark at **exactly the third lost leg** — the
step where the GF(2) brute force proves recovery impossible. Heal the boundary and the
bulk comes back.

## The arc, closed

Research 44/46 built the dictionary; research 48 made it exact; this set explains it:
entanglement entropy is a flow you can route (Part I), geometry is a renormalization
network you can count (Part II), and the bulk is the protected logical subspace of a
code you can break and heal (Part III). Spacetime doesn't just *have* quantum
information — structurally, it *is* quantum information.

## Sources

- M. Freedman, M. Headrick, *Bit threads and holographic entanglement*,
  [1604.00354](https://arxiv.org/abs/1604.00354)
- B. Swingle, *Entanglement Renormalization and Holography*,
  [0905.1317](https://arxiv.org/abs/0905.1317)
- A. Almheiri, X. Dong, D. Harlow, *Bulk Locality and Quantum Error Correction in
  AdS/CFT*, [1411.7041](https://arxiv.org/abs/1411.7041)
- F. Pastawski, B. Yoshida, D. Harlow, J. Preskill, *Holographic quantum
  error-correcting codes: Toy models for the bulk/boundary correspondence*,
  [1503.06237](https://arxiv.org/abs/1503.06237)
- R. Laflamme, C. Miquel, J.-P. Paz, W. Zurek, *Perfect Quantum Error Correction Code*,
  [quant-ph/9602019](https://arxiv.org/abs/quant-ph/9602019)
- T. Faulkner, M. Guica, T. Hartman, R. Myers, M. Van Raamsdonk, *Gravitation from
  Entanglement in Holographic CFTs*, [1312.7856](https://arxiv.org/abs/1312.7856)
