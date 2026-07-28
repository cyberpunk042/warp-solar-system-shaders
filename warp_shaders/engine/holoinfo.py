"""Holographic quantum information (host-side): bit threads, MERA, and the code.

The engine core behind the quantum-information set — the HOW behind research 46's claim
that entanglement builds geometry:

* ``five_qubit_stabilizers`` / ``erasure_correctable`` — the **[[5,1,3]] perfect code**
  (Laflamme et al. 1996), implemented as actual stabilizers over GF(2): one logical
  qubit in five, distance 3. ``erasure_correctable`` brute-force checks whether any
  logical operator is supported on the erased set — the exact criterion. The perfect
  code is quantum-MDS: ANY two erasures are correctable, ANY three are not (no-cloning:
  the two survivors would otherwise hold a copy). This is the seed tensor of the HaPPY
  holographic code.
* ``build_disk_graph`` / ``max_flow`` — a hyperbolic polar grid on the Poincaré disk
  with edge capacities equal to the hyperbolic length of the crossed dual segment, plus
  Dinic's algorithm. **Max flow from a boundary interval to its complement = the min
  cut = the RT geodesic length** (Freedman-Headrick bit threads / max-flow-min-cut):
  entanglement entropy as the maximum number of independent threads you can pull out of
  the region. The suite asserts flow == cut exactly (MFMC) and that the flow tracks the
  analytic geodesic length ``2·ln(sin(Δθ/2)) + const`` across interval sizes.
* ``mera_layers`` / ``mera_cut_bonds`` — the ideal binary MERA count: an interval of ℓ
  sites is coarse-grained away in ``ceil(log₂ ℓ)`` layers, and the minimal cut through
  the network severs ``~2·log₂ ℓ + O(1)`` bonds — the network-level origin of the CFT
  log law S = (c/3)·ln ℓ (Swingle: MERA is a discretized AdS time slice).
* ``happy_erased_legs`` / ``happy_central_recoverable`` — the HaPPY pentagon code's
  central tensor: five legs at angles 2πj/5; a contiguous erased boundary arc kills the
  legs it covers, and the central logical qubit survives iff at least 3 of 5 legs remain
  — which the suite CROSS-CHECKS against ``erasure_correctable`` on the actual [[5,1,3]]
  code: the geometric wedge rule and the algebraic code rule are the same fact.

See ``docs/research/49-holographic-quantum-information.md``.
"""

import math

# ---------------------------------------------------------------------------
# The [[5,1,3]] perfect code, as honest stabilizers over GF(2)
# ---------------------------------------------------------------------------

def five_qubit_stabilizers():
    """The four stabilizer generators of the [[5,1,3]] code, cyclic XZZXI, as
    ``(x_mask, z_mask)`` bit-mask pairs (qubit j ↔ bit j)."""
    gens = []
    base = "XZZXI"
    for s in range(4):
        x = z = 0
        for j in range(5):
            p = base[(j - s) % 5]
            if p == "X":
                x |= 1 << j
            elif p == "Z":
                z |= 1 << j
        gens.append((x, z))
    return gens


def _commutes(a, b):
    """Symplectic form over GF(2): Paulis (xa,za),(xb,zb) commute iff
    |xa·zb| + |za·xb| is even."""
    return (bin(a[0] & b[1]).count("1") + bin(a[1] & b[0]).count("1")) % 2 == 0


def _in_span(vec, basis):
    """Is the 10-bit symplectic vector in the GF(2) span of ``basis``? (Gaussian
    elimination over Z2.)"""
    rows = list(basis)
    v = vec
    for _ in range(len(rows)):
        piv = max((r for r in rows if r), default=0)
        if piv == 0:
            break
        hi = piv.bit_length() - 1
        rows = [r ^ piv if r != piv and (r >> hi) & 1 else r for r in rows if r != piv]
        if (v >> hi) & 1:
            v ^= piv
    return v == 0


def erasure_correctable(erased):
    """Can the [[5,1,3]] code correct erasure of the qubit set ``erased``?

    Exact criterion: an erasure is correctable iff NO logical operator (an element of
    the normalizer that is not a stabilizer) is supported entirely on the erased set.
    Brute-forced over all 4^|E| Paulis on E — five qubits keep this tiny."""
    erased = sorted(set(erased))
    gens = five_qubit_stabilizers()
    sym = [(x << 5) | z for (x, z) in gens]
    for pat in range(4 ** len(erased)):
        x = z = 0
        p = pat
        for q in erased:
            k = p % 4
            p //= 4
            if k in (1, 3):
                x |= 1 << q
            if k in (2, 3):
                z |= 1 << q
        if x == 0 and z == 0:
            continue
        if all(_commutes((x, z), g) for g in gens) and not _in_span((x << 5) | z, sym):
            return False                      # a logical operator lives on the erasure
    return True


# ---------------------------------------------------------------------------
# Bit threads: hyperbolic grid + Dinic max-flow (= min cut = RT geodesic)
# ---------------------------------------------------------------------------

class _Dinic:
    def __init__(self, n):
        self.n = n
        self.adj = [[] for _ in range(n)]

    def add_edge(self, u, v, cap):
        self.adj[u].append([v, cap, len(self.adj[v])])
        self.adj[v].append([u, cap, len(self.adj[u]) - 1])   # undirected: cap both ways

    def _bfs(self, s, t):
        self.level = [-1] * self.n
        self.level[s] = 0
        q = [s]
        for u in q:
            for e in self.adj[u]:
                if e[1] > 1e-12 and self.level[e[0]] < 0:
                    self.level[e[0]] = self.level[u] + 1
                    q.append(e[0])
        return self.level[t] >= 0

    def _dfs(self, u, t, f):
        if u == t:
            return f
        while self.it[u] < len(self.adj[u]):
            e = self.adj[u][self.it[u]]
            if e[1] > 1e-12 and self.level[e[0]] == self.level[u] + 1:
                d = self._dfs(e[0], t, min(f, e[1]))
                if d > 1e-12:
                    e[1] -= d
                    self.adj[e[0]][e[2]][1] += d
                    return d
            self.it[u] += 1
        return 0.0

    def max_flow(self, s, t):
        flow = 0.0
        while self._bfs(s, t):
            self.it = [0] * self.n
            while True:
                f = self._dfs(s, t, float("inf"))
                if f <= 1e-12:
                    break
                flow += f
        return flow

    def min_cut_capacity(self, s, original_caps):
        """Capacity of the s-side cut in the residual graph (should equal max flow)."""
        seen = [False] * self.n
        seen[s] = True
        q = [s]
        for u in q:
            for e in self.adj[u]:
                if e[1] > 1e-12 and not seen[e[0]]:
                    seen[e[0]] = True
                    q.append(e[0])
        cut = 0.0
        for (u, v, c) in original_caps:
            if seen[u] != seen[v]:
                cut += c
        return cut


def build_disk_graph(n_rings=24, n_sectors=96, u_max=4.0):
    """A polar grid on the Poincaré disk with HYPERBOLIC capacities.

    Nodes at (u_i, φ_j), u uniform in (0, u_max] (hyperbolic radius, r = tanh(u/2)).
    Each edge's capacity is the hyperbolic length of the dual segment it crosses, so a
    graph cut measures hyperbolic length and the min cut converges to the RT geodesic.
    Returns ``(node_count, edges, boundary_ring)`` with edges as (u, v, cap) triples and
    node id = 1 + (i·n_sectors + j) (node 0 is reserved: the disk centre hub)."""
    du = u_max / n_rings
    edges = []
    def nid(i, j):
        return 1 + i * n_sectors + (j % n_sectors)
    for i in range(n_rings):
        u_i = (i + 1) * du
        # angular edges along ring i: cross a radial dual segment of hyperbolic length du
        for j in range(n_sectors):
            edges.append((nid(i, j), nid(i, j + 1), du))
        # radial edges from ring i-1 to i: cross an angular dual arc at u_mid
        u_mid = u_i - 0.5 * du
        arc = math.sinh(u_mid) * (2.0 * math.pi / n_sectors)
        for j in range(n_sectors):
            inner = 0 if i == 0 else nid(i - 1, j)
            edges.append((inner, nid(i, j), arc))
    return 1 + n_rings * n_sectors, edges, n_rings - 1


def interval_max_flow(dtheta, n_rings=24, n_sectors=96, u_max=4.0):
    """Max flow (= min cut) from a boundary interval of angular size ``dtheta`` to its
    complement, on the hyperbolic grid. Freedman-Headrick: this IS the entanglement
    entropy in length units (4G = 1) up to the grid's UV offset. Returns
    ``(flow, cut_capacity)`` — MFMC guarantees they are equal."""
    n, edges, outer = build_disk_graph(n_rings, n_sectors, u_max)
    src, snk = n, n + 1
    g = _Dinic(n + 2)
    caps = []
    for (u, v, c) in edges:
        g.add_edge(u, v, c)
        caps.append((u, v, c))
    big = 1e9
    for j in range(n_sectors):
        th = (j + 0.5) * 2.0 * math.pi / n_sectors
        node = 1 + outer * n_sectors + j
        if min(th, 2.0 * math.pi - th) * 2.0 <= dtheta or th <= 0.5 * dtheta \
                or th >= 2.0 * math.pi - 0.5 * dtheta:
            g.adj[src].append([node, big, len(g.adj[node])])
            g.adj[node].append([src, 0.0, len(g.adj[src]) - 1])
        else:
            g.adj[node].append([snk, big, len(g.adj[snk])])
            g.adj[snk].append([node, 0.0, len(g.adj[node]) - 1])
    flow = g.max_flow(src, snk)
    return flow, g.min_cut_capacity(src, caps)


# ---------------------------------------------------------------------------
# MERA: the network origin of the log law
# ---------------------------------------------------------------------------

def mera_layers(l_sites):
    """Layers of a binary MERA needed to coarse-grain an interval of ``l_sites`` away:
    ``ceil(log2 l)`` — the RG depth of the interval (its causal-cone height)."""
    return max(0, math.ceil(math.log2(max(l_sites, 1))))


def mera_cut_bonds(l_sites, c0=2):
    """Bonds severed by the minimal cut through the ideal binary MERA around an
    interval of ``l_sites``: the cut climbs one layer per halving on each side —
    ``2·ceil(log2 l) + c0``. With bond dimension χ each bond carries ≤ ln χ, giving
    ``S ≤ (2·log2 l + c0)·ln χ`` — the network's log law, Swingle's discrete RT."""
    return 2 * mera_layers(l_sites) + c0


# ---------------------------------------------------------------------------
# HaPPY: the pentagon code's central qubit vs an erased boundary arc
# ---------------------------------------------------------------------------

def happy_erased_legs(erased_arc, offset=0.0):
    """Which of the central pentagon's five legs (at angles 2πj/5 + offset) fall inside
    a contiguous erased boundary arc ``[0, erased_arc)`` (radians)? Returns the tuple of
    erased leg indices."""
    out = []
    for j in range(5):
        a = (2.0 * math.pi * j / 5.0 + offset) % (2.0 * math.pi)
        if a < erased_arc:
            out.append(j)
    return tuple(out)


def happy_central_recoverable(erased_arc, offset=0.0):
    """Is the HaPPY code's CENTRAL logical qubit still reconstructable from the intact
    boundary? Geometric rule: yes iff at least 3 of its 5 legs survive — which is the
    [[5,1,3]] erasure criterion applied to the seed tensor (the suite cross-checks the
    two). Returns ``(recoverable, n_erased_legs)``."""
    erased = happy_erased_legs(erased_arc, offset)
    return (len(erased) <= 2, len(erased))
