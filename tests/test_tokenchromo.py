"""Tests for C3 tokenize->web->DNA->chromosome (warp_compress.tokenchromo) — the card as a coiled genome.

Operator spec: break the item into a web of tokens (one word per atom), read them as a DNA-equivalent
sequence, and compress that through the whole chromosome (coil) process.

  1. exact lossless round-trip: uncoil -> untokenize -> the exact card;
  2. the coil actually happens — repeated token-PHRASES wrap into nucleosome rules, many layers deep;
  3. the chromosome beats the flat C1 index (the sequence-level coil earns extra compression);
  4. a repetitive genome condenses far more than a random one (the coil exploits real structure).

    python -m tests.test_tokenchromo
"""

import numpy as np

from warp_compress import tokenchromo as tc
from warp_compress.chromosome import coil, uncoil


def main():
    # 1-3. on the real board
    r = tc.compress_card(block=4)
    assert r["lossless"], "tokenize->chromosome round-trip not lossless"
    print(f"  lossless round-trip: OK  (card exact; {r['atoms']} atoms, {r['vocab']} words)")

    assert r["nucleosomes"] > 0 and r["layers"] >= 2, \
        f"no real coiling (nucleosomes {r['nucleosomes']}, layers {r['layers']})"
    print(f"  chromosome coils: OK  ({r['nucleosomes']} nucleosomes, {r['layers']} layers deep)")

    assert r["ratio"] > 1.0, f"did not compress (ratio {r['ratio']:.2f})"
    assert r["ratio_vs_c1"] > 1.0, f"chromosome coil did not beat flat C1 ({r['ratio_vs_c1']:.2f})"
    print(f"  compresses + beats flat C1: OK  ({r['ratio']:.1f}x total, {r['ratio_vs_c1']:.2f}x vs C1)")

    # 4. a repetitive genome condenses far more than a random one
    rng = np.random.default_rng(0)
    motif = rng.integers(0, 8, 12).tolist()
    repetitive = motif * 40                          # the same phrase, over and over
    randseq = rng.integers(0, 8, len(repetitive)).tolist()
    cr = coil(repetitive, base=8)
    cn = coil(randseq, base=8)
    assert uncoil(cr) == repetitive and uncoil(cn) == randseq, "coil round-trip failed"
    rep_size = 2 * len(cr.rules) + len(cr.top)
    rnd_size = 2 * len(cn.rules) + len(cn.top)
    assert rep_size < rnd_size, f"repetition not exploited (rep {rep_size} vs random {rnd_size})"
    print(f"  repetition condenses: OK  (repetitive -> {rep_size} symbols, random -> {rnd_size})")

    # 5. the warp_tokenize_chromo scene (C3 as a process) renders on the real board and animates
    #    across card -> web of token-words -> DNA helix -> chromosome
    import warp as wp
    import warp_shaders as ws
    wp.init()
    card = np.asarray(ws.render("warp_tokenize_chromo", width=120, height=90, time=0.6), np.float32)   # card+scan
    helix = np.asarray(ws.render("warp_tokenize_chromo", width=120, height=90, time=7.0), np.float32)  # DNA helix
    assert np.all(np.isfinite(card)) and card.max() > 0.1 and card.std() > 0.01, "tokenize_chromo: bad frame"
    assert np.abs(card - helix).mean() > 1e-3, "tokenize_chromo: card -> DNA helix did not animate"
    print("  scene warp_tokenize_chromo: OK")

    print("ALL PASSED")


if __name__ == "__main__":
    main()
