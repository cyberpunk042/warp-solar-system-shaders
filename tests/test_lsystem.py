"""Self-tests for the L-System core — one grammar per class, verified against
known expansions from ABOP (Prusinkiewicz & Lindenmayer, 1990).

Run: `python -m tests.test_lsystem` (or under pytest).
"""

from warp_shaders.life.lsystem import LSystem, Module, Rule, parse, word_to_str


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


if __name__ == "__main__":
    print("L-System core tests:")
    test_parse_roundtrip(); print("  parse round-trip: OK")
    test_d0l_algae(); print("  D0L algae (Fibonacci lengths): OK")
    test_d0l_koch(); print("  D0L Koch curve: OK")
    test_d0l_dragon(); print("  D0L dragon curve: OK")
    test_stochastic_reproducible(); print("  stochastic (seeded, reproducible): OK")
    test_parametric(); print("  parametric (param arithmetic): OK")
    test_parametric_condition(); print("  parametric (conditional): OK")
    test_context_sensitive_signal(); print("  context-sensitive (signal propagation): OK")
    test_context_bracket_aware(); print("  context (bracket-aware neighbours): OK")
    print("ALL PASSED")
