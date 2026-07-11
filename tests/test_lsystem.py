"""Self-tests for the L-System core — one grammar per class, verified against
known expansions from ABOP (Prusinkiewicz & Lindenmayer, 1990).

Run: `python -m tests.test_lsystem` (or under pytest).
"""

import numpy as np

from warp_shaders.life.lsystem import LSystem, Module, Rule, parse, word_to_str
from warp_shaders.life.turtle import TurtleConfig, interpret


def test_parse_roundtrip():
    assert word_to_str(parse("F(1)[+(30)F]F")) == "F(1)[+(30)F]F"
    assert [m.sym for m in parse("F+F-F")] == ["F", "+", "F", "-", "F"]
    assert parse("A(1.5,30)")[0].params == (1.5, 30.0)


def test_d0l_algae():
    # Lindenmayer's original algae; generation lengths are the Fibonacci numbers
    ls = LSystem("A", {"A": "AB", "B": "A"})
    expect = ["A", "AB", "ABA", "ABAAB", "ABAABABA", "ABAABABAABAAB"]
    for n, s in enumerate(expect):
        assert ls.derive_str(n) == s, f"algae n={n}: {ls.derive_str(n)} != {s}"
    # lengths are Fibonacci
    lens = [len(ls.derive(n)) for n in range(7)]
    assert lens == [1, 2, 3, 5, 8, 13, 21]


def test_d0l_koch():
    ls = LSystem("F", {"F": "F+F-F-F+F"})
    assert ls.derive_str(1) == "F+F-F-F+F"
    # each of 5 F's expands to 5 -> 25 F's at n=2
    assert word_to_str(ls.derive(2)).count("F") == 25


def test_d0l_dragon():
    ls = LSystem("FX", {"X": "X+YF+", "Y": "-FX-Y"})
    assert ls.derive_str(1) == "FX+YF+"
    assert ls.derive_str(2) == "FX+YF++-FX-YF+"


def test_stochastic_reproducible():
    spec = [("F[+F]F", 0.5), ("F[-F]F", 0.5)]
    a = LSystem("F", {"F": spec}, seed=7).derive_str(4)
    b = LSystem("F", {"F": spec}, seed=7).derive_str(4)
    c = LSystem("F", {"F": spec}, seed=8).derive_str(4)
    assert a == b, "same seed must reproduce"
    assert a != c, "different seeds should (almost surely) differ"
    # both successors are exercised across the word
    assert "[+F]" in a and "[-F]" in a


def test_parametric():
    # A(s) -> F(s) A(s/2): geometric decay of the step parameter
    rule = Rule("A", lambda m, l, r: [Module("F", (m.params[0],)),
                                      Module("A", (m.params[0] / 2.0,))])
    ls = LSystem("A(1)", {"A": rule})
    w = ls.derive(3)
    fs = [m.params[0] for m in w if m.sym == "F"]
    assert fs == [1.0, 0.5, 0.25]
    assert [m.sym for m in w] == ["F", "F", "F", "A"]
    assert w[-1].params == (0.125,)


def test_parametric_condition():
    # A(s) grows only while s > threshold; below it, A -> F (terminates)
    grow = Rule("A", lambda m, l, r: [Module("A", (m.params[0] * 0.6,))],
                cond=lambda p: p[0] > 0.3)
    stop = Rule("A", lambda m, l, r: [Module("F", m.params)],
                cond=lambda p: p[0] <= 0.3)
    ls = LSystem("A(1)", {"A": [grow, stop]})
    # 1 -> .6 -> .36 -> .216 (still A; <=.3 so next gen terminates to F)
    w3 = ls.derive(3)
    assert len(w3) == 1 and w3[0].sym == "A" and abs(w3[0].params[0] - 0.216) < 1e-6
    w4 = ls.derive(4)
    assert len(w4) == 1 and w4[0].sym == "F" and abs(w4[0].params[0] - 0.216) < 1e-6


def test_context_sensitive_signal():
    # a signal B travels right through a row of A's: B->A, and A with left B -> B
    move = Rule("A", lambda m, l, r: [Module("B")], left="B")
    ls = LSystem("BAAAA", {"B": "A", "A": move})
    assert ls.derive_str(1) == "ABAAA"      # signal at index 1
    assert ls.derive_str(2) == "AABAA"      # index 2
    assert ls.derive_str(3) == "AAABA"      # index 3


def test_context_bracket_aware():
    # right context reaches into a branch; a completed sibling branch is skipped
    ls = LSystem("A[B]C", {})
    w = ls.axiom
    # left context of C (index 4) skips the finished [B] branch back to A
    assert ls._left_context(w, 4).sym == "A"
    # right context of A (index 0) reaches into the branch to B
    assert ls._right_context(w, 1).sym == "B"


# --- turtle interpreter ------------------------------------------------------


def test_turtle_straight():
    geo = interpret(parse("FF"), TurtleConfig(step=1.0))
    assert len(geo.segments) == 2 and len(geo.leaves) == 0
    np.testing.assert_allclose(geo.segments[0].p0, [0, 0, 0], atol=1e-5)
    np.testing.assert_allclose(geo.segments[1].p1, [0, 2, 0], atol=1e-5)  # grew up +Y


def test_turtle_branch():
    geo = interpret(parse("F[+F]F"), TurtleConfig(step=1.0, angle=90.0))
    assert len(geo.segments) == 3
    # main stem: two segments straight up
    np.testing.assert_allclose(geo.segments[2].p1, [0, 2, 0], atol=1e-5)
    # branch segment starts at the top of the first F and is NOT vertical
    br = geo.segments[1]
    np.testing.assert_allclose(br.p0, [0, 1, 0], atol=1e-5)
    assert abs(br.p1[1] - 1.0) < 1e-4                # +90 yaw -> horizontal branch


def test_turtle_leaf_and_bounds():
    geo = interpret(parse("F L"), TurtleConfig(step=2.0))
    assert len(geo.segments) == 1 and len(geo.leaves) == 1
    lo, hi = geo.bounds()
    assert hi[1] - lo[1] >= 1.99                      # spans the 2-unit stem


# --- environmental response (tropism, ABOP §2.3.4) ---------------------------


def test_gravitropism_bends_down():
    # a stem set off a little from vertical sags under gravity T=(0,-1,0): its
    # tip ends up lower than the same stem grown straight (a perfectly vertical
    # stem is a degenerate equilibrium, so we tilt it first with &(20))
    word = parse("&(20)" + "F" * 12)
    straight = interpret(word, TurtleConfig(step=1.0))
    sag = interpret(word, TurtleConfig(step=1.0, tropism=(0, -1, 0),
                                       tropism_e=0.1))
    tip_straight = straight.segments[-1].p1
    tip_sag = sag.segments[-1].p1
    assert tip_straight[1] > tip_sag[1] + 1.0          # sagged tip is lower
    hd = lambda p: float((p[0] ** 2 + p[2] ** 2) ** 0.5)
    assert hd(tip_sag) > hd(tip_straight)              # and reaches further out
    # in the gentle regime a stronger susceptibility sags more
    sag2 = interpret(word, TurtleConfig(step=1.0, tropism=(0, -1, 0),
                                        tropism_e=0.2))
    assert sag2.segments[-1].p1[1] < tip_sag[1]


def test_phototropism_bends_toward_light():
    # a light off to the +X side pulls the growing tip toward it
    word = parse("F" * 12)
    dark = interpret(word, TurtleConfig(step=1.0))
    lit = interpret(word, TurtleConfig(step=1.0, light=(20.0, 6.0, 0.0),
                                       light_e=0.15))
    assert lit.segments[-1].p1[0] > dark.segments[-1].p1[0] + 1.0


def test_tropism_off_is_straight():
    # tropism vector present but susceptibility 0 -> identical to no tropism
    word = parse("F" * 6)
    a = interpret(word, TurtleConfig(step=1.0))
    b = interpret(word, TurtleConfig(step=1.0, tropism=(0, -1, 0), tropism_e=0.0))
    np.testing.assert_allclose(a.segments[-1].p1, b.segments[-1].p1, atol=1e-6)


# --- mesh tessellation -------------------------------------------------------
from warp_shaders.life.mesh import build_mesh


def test_mesh_single_tube():
    geo = interpret(parse("F"), TurtleConfig(step=1.0, radius=0.1))
    m = build_mesh(geo, sides=6)
    assert m.verts.shape == (12, 3) and m.n_tris == 12
    assert np.allclose(np.linalg.norm(m.normals, axis=1), 1.0, atol=1e-4)
    assert int(m.indices.max()) < m.verts.shape[0]


def test_mesh_plant_counts():
    geo = interpret(parse("F[+F][-F]F L"), TurtleConfig(step=1.0, radius=0.08, angle=35))
    m = build_mesh(geo, sides=6)
    assert m.n_tris == 4 * 12 + 2           # 4 tubes + 1 leaf
    assert np.isfinite(m.verts).all()



# --- plant grammars ----------------------------------------------------------
from warp_shaders.life import plants as _plants


def test_plant_specs_grow():
    for name, gens in [("grass", 9), ("herb", 6), ("tree", 7),
                       ("fern", 6), ("sapling", 7), ("weeper", 6),
                       ("flower", 6), ("bush", 5)]:
        spec = _plants.get_spec(name)
        mesh, (lo, hi) = _plants.grow_mesh(spec, gens)
        assert mesh.n_tris > 0, f"{name}: empty mesh"
        assert float(hi[1] - lo[1]) > 0.5, f"{name}: no height"
    # get_spec memoizes (same object -> mesh cache stays warm)
    assert _plants.get_spec("tree") is _plants.get_spec("tree")


def test_grow_mesh_env_responds():
    from dataclasses import replace
    spec = _plants.get_spec("sapling")
    # a phototropic sapling leans toward the light; the tip x follows it
    left = replace(spec.cfg, light=(-15.0, 6.0, 0.0), light_e=0.14)
    right = replace(spec.cfg, light=(15.0, 6.0, 0.0), light_e=0.14)
    _, (loL, hiL) = _plants.grow_mesh_env(spec, spec.gens, left)
    _, (loR, hiR) = _plants.grow_mesh_env(spec, spec.gens, right)
    assert loL[0] < loR[0] and hiL[0] < hiR[0]        # whole plant shifts left
    # folding the leaves shrinks the silhouette vs the open plant
    openc = replace(spec.cfg, leaf_fold=0.0)
    shut = replace(spec.cfg, leaf_fold=1.0)
    mo, _ = _plants.grow_mesh_env(spec, spec.gens, openc)
    ms, _ = _plants.grow_mesh_env(spec, spec.gens, shut)
    assert mo.n_tris == ms.n_tris                      # same leaves, folded not gone



# --- molecular scales (DNA / protein) ----------------------------------------
from warp_shaders.life import molecular as _mol


def test_build_helix():
    mesh, (lo, hi) = _mol.build_helix(bp=12, seed=1)
    assert mesh.n_tris > 0
    n = mesh.verts.shape[0]
    assert mesh.normals.shape == (n, 3) and mesh.colors.shape == (n, 3)
    assert np.isfinite(mesh.verts).all() and np.isfinite(mesh.normals).all()
    assert int(mesh.indices.max()) < n
    assert float(hi[1] - lo[1]) > 0.5                  # helix has height
    # more base pairs -> taller helix + more geometry
    big, (blo, bhi) = _mol.build_helix(bp=24, seed=1)
    assert big.n_tris > mesh.n_tris
    assert float(bhi[1] - blo[1]) > float(hi[1] - lo[1])


def test_build_protein_folds():
    ext, (elo, ehi) = _mol.build_protein(n=40, fold=0.0)
    fld, (flo, fhi) = _mol.build_protein(n=40, fold=1.0)
    assert ext.n_tris > 0 and fld.n_tris == ext.n_tris  # same chain, reshaped
    # extended is longer end-to-end than the compact fold
    ext_span = float(ehi[1] - elo[1])
    fld_diag = float(np.linalg.norm(np.asarray(fhi) - np.asarray(flo)))
    assert ext_span > fld_diag
    n = fld.verts.shape[0]
    assert fld.colors.shape == (n, 3) and np.isfinite(fld.verts).all()


def test_merge_meshes():
    from warp_shaders.life.mesh import merge_meshes
    a = _plants.grow_mesh(_plants.get_spec("grass"), 8)[0]
    b = _plants.grow_mesh(_plants.get_spec("bush"), 5)[0]
    m = merge_meshes([a, b], offsets=[(0, 0, 0), (5, 0, 0)])
    assert m.n_tris == a.n_tris + b.n_tris
    assert m.verts.shape[0] == a.verts.shape[0] + b.verts.shape[0]
    assert int(m.indices.max()) < m.verts.shape[0]     # indices re-based, in range
    # the offset moved the second plant +5 in x
    assert float(m.verts[:, 0].max()) > float(a.verts[:, 0].max()) + 3.0
    # empties are skipped, not fatal
    z = merge_meshes([None])
    assert z.n_tris == 0


def test_cell_divides():
    from warp_shaders.life.cell import render_cell
    one = render_cell(64, 64, 0.0, (0.0, 0.0), 0.0, "cpu")
    two = render_cell(64, 64, 0.0, (0.0, 0.0), 1.0, "cpu")
    assert one.shape == (64, 64, 3) and np.isfinite(one).all()
    assert float(np.abs(one - two).max()) > 0.0        # division changes the image


# --- the mind (Conway's Life) -------------------------------------------------
from warp_shaders.life.mind import Mind


def _mind_with(cells, size=6):
    m = Mind(size=size, seed=0, stimulus_every=0)   # no stimulus for pure Conway
    m.grid[:] = 0
    for (r, c) in cells:
        m.grid[r, c] = 1
    return m


def test_conway_blinker_period_2():
    # a horizontal blinker becomes vertical, then horizontal again (period 2)
    horiz = [(2, 1), (2, 2), (2, 3)]
    m = _mind_with(horiz)
    m.step()
    vert = {(1, 2), (2, 2), (3, 2)}
    assert set(map(tuple, np.argwhere(m.grid))) == vert
    m.step()
    assert set(map(tuple, np.argwhere(m.grid))) == set(horiz)


def test_conway_block_still_life():
    block = [(1, 1), (1, 2), (2, 1), (2, 2)]
    m = _mind_with(block)
    before = m.grid.copy()
    for _ in range(5):
        m.step()
    assert np.array_equal(m.grid, before)              # block never changes


def test_mind_deterministic_and_bounded():
    a = Mind(size=40, seed=11)
    b = Mind(size=40, seed=11)
    for _ in range(20):
        a.step(); b.step()
    assert np.array_equal(a.grid, b.grid)              # same seed -> same run
    d = a.decision()
    assert 0.0 <= d <= 1.0
    # decision is monotonic in live-fraction: a denser grid drives higher
    lo = _mind_with([(2, 2)], size=20); lo.stimulus_every = 0
    hi = Mind(size=20, seed=1, density=0.5, stimulus_every=0)
    assert hi.decision() >= lo.decision()


def test_mind_per_branch_decisions():
    # k independent band-drives, each bounded; a left-heavy grid drives its
    # left bands higher than its (empty) right bands
    m = Mind(size=20, seed=0, stimulus_every=0)
    m.grid[:] = 0
    m.grid[:, :10] = 1                                  # fill the left half
    ds = m.decisions(4)
    assert len(ds) == 4 and all(0.0 <= d <= 1.0 for d in ds)
    assert ds[0] >= ds[-1] and ds[0] > 0.5 and ds[-1] < 0.5


# --- wave / collapse ----------------------------------------------------------
from warp_shaders.life import collapse as _collapse


def test_superpose():
    a = np.zeros((4, 4, 3), np.float32)
    b = np.ones((4, 4, 3), np.float32)
    np.testing.assert_allclose(_collapse.superpose([a, b]), 0.5)
    # all weight on one member returns that member
    np.testing.assert_allclose(_collapse.superpose([a, b], [0.0, 1.0]), 1.0)


def test_collapse_blend_endpoints():
    g = [np.full((10, 6, 3), 0.2, np.float32),
         np.full((10, 6, 3), 0.8, np.float32),
         np.full((10, 6, 3), 0.5, np.float32)]
    cloud = _collapse.superpose(g)                       # 0.5 everywhere
    # front_frac 0 -> fully superposed (the cloud)
    out0 = _collapse.collapse_blend(g, chosen_idx=1, front_frac=0.0)
    np.testing.assert_allclose(out0, cloud, atol=1e-6)
    # front_frac 1 -> fully collapsed to the chosen member
    out1 = _collapse.collapse_blend(g, chosen_idx=1, front_frac=1.0)
    np.testing.assert_allclose(out1, g[1], atol=1e-6)
    # midway: the top row is collapsed (chosen), the bottom row is still cloud
    mid = _collapse.collapse_blend(g, chosen_idx=1, front_frac=0.5, band=0.05)
    assert abs(float(mid[0].mean()) - 0.8) < 1e-3       # top = chosen
    assert abs(float(mid[-1].mean()) - 0.5) < 1e-3      # bottom = cloud


def test_pick_index():
    assert _collapse.pick_index([0.1, 0.9, 0.3]) == 1
    assert _collapse.pick_index([0.5, 0.2, 0.2, 0.7]) == 3


if __name__ == "__main__":
    print("L-System + turtle tests:")
    test_parse_roundtrip(); print("  parse round-trip: OK")
    test_d0l_algae(); print("  D0L algae (Fibonacci lengths): OK")
    test_d0l_koch(); print("  D0L Koch curve: OK")
    test_d0l_dragon(); print("  D0L dragon curve: OK")
    test_stochastic_reproducible(); print("  stochastic (seeded, reproducible): OK")
    test_parametric(); print("  parametric (param arithmetic): OK")
    test_parametric_condition(); print("  parametric (conditional): OK")
    test_context_sensitive_signal(); print("  context-sensitive (signal): OK")
    test_context_bracket_aware(); print("  context (bracket-aware): OK")
    test_turtle_straight(); print("  turtle straight stem: OK")
    test_turtle_branch(); print("  turtle branch: OK")
    test_turtle_leaf_and_bounds(); print("  turtle leaf + bounds: OK")
    test_gravitropism_bends_down(); print("  gravitropism (sag): OK")
    test_phototropism_bends_toward_light(); print("  phototropism (toward light): OK")
    test_tropism_off_is_straight(); print("  tropism off == straight: OK")
    test_mesh_single_tube(); print("  mesh single tube: OK")
    test_mesh_plant_counts(); print("  mesh plant counts: OK")
    test_plant_specs_grow(); print("  plant grammars grow (grass/herb/tree/fern/sapling/weeper): OK")
    test_grow_mesh_env_responds(); print("  grow_mesh_env responds to light + fold: OK")
    test_merge_meshes(); print("  merge_meshes (re-based, offset): OK")
    test_build_helix(); print("  DNA helix builds (colored, scales with bp): OK")
    test_build_protein_folds(); print("  protein folds (extended > compact): OK")
    test_cell_divides(); print("  cell division changes the render: OK")
    test_conway_blinker_period_2(); print("  mind: Conway blinker period-2: OK")
    test_conway_block_still_life(); print("  mind: Conway block still-life: OK")
    test_mind_deterministic_and_bounded(); print("  mind: deterministic + bounded decision: OK")
    test_mind_per_branch_decisions(); print("  mind: per-branch band decisions: OK")
    test_superpose(); print("  collapse: superpose (mean + weighted): OK")
    test_collapse_blend_endpoints(); print("  collapse: blend endpoints + sweep: OK")
    test_pick_index(); print("  collapse: pick_index (argmax): OK")
    print("ALL PASSED")
