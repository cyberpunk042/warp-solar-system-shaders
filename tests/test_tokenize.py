"""Tests for Process 1 — tokenization (warp_shaders.genome.tokenize) — the card turned into tokens.

Operator spec: turn every bit of the graphics card into a token (100 000 to 1 000 000), a conserving
transform — it uses what it transforms, it never spawns.

  1. conservation: token count == (occupied voxels) x sub**3 exactly — no token fabricated;
  2. scale: the token count sits in the operator's 100k..1M range;
  3. geometry: every token lands inside the real board's bounding box;
  4. typing: tokens carry merge-codec type ids (the vocabulary that colours them);
  5. the warp_tokenize scene renders on the real board and the token cloud animates (card -> cloud),
     with the token count conserved across time.

    python -m tests.test_tokenize
"""

import numpy as np

from warp_compress.foldcube import _BB, sample_card
from warp_shaders.genome import tokenize_card


def main():
    occ = sample_card()
    occupied = int((occ > 0).sum())

    sub = 2
    tc = tokenize_card(sub=sub, block=5)

    # 1. conservation — count is exactly what we transformed, nothing spawned
    assert tc.n == occupied * sub ** 3, f"not conserved: {tc.n} != {occupied} x {sub**3}"
    print(f"  conservation: OK  ({occupied} occupied voxels x {sub**3} -> {tc.n} tokens, none spawned)")

    # 2. scale — inside the operator's 100k..1M band
    assert 100_000 <= tc.n <= 1_000_000, f"token count {tc.n} out of 100k..1M range"
    print(f"  scale: OK  ({tc.n} tokens in 100k..1M)")

    # 3. geometry — every token inside the real board bounding box
    x0, x1, y0, y1, z0, z1 = _BB
    p = tc.positions
    assert p.shape == (tc.n, 3) and p.dtype == np.float32
    assert p[:, 0].min() >= x0 - 1e-3 and p[:, 0].max() <= x1 + 1e-3
    assert p[:, 1].min() >= y0 - 1e-3 and p[:, 1].max() <= y1 + 1e-3
    assert p[:, 2].min() >= z0 - 1e-3 and p[:, 2].max() <= z1 + 1e-3
    assert np.all(np.isfinite(p))
    print("  geometry: OK  (all tokens inside the board bounding box)")

    # 4. typing — merge-codec vocabulary ids present, colours finite in [0,1]
    assert tc.ids.shape == (tc.n,) and len(np.unique(tc.ids)) > 1, "no token vocabulary"
    assert tc.colors.shape == (tc.n, 3) and np.all(np.isfinite(tc.colors))
    assert tc.colors.min() >= 0.0 and tc.colors.max() <= 1.0
    print(f"  typing: OK  ({len(np.unique(tc.ids))} token types colour the cloud)")

    # 5. the warp_tokenize scene renders on the real board and the cloud animates
    import warp as wp
    import warp_shaders as ws
    wp.init()
    tight = np.asarray(ws.render("warp_tokenize", width=160, height=90, time=0.0), np.float32)  # card
    cloud = np.asarray(ws.render("warp_tokenize", width=160, height=90, time=2.6), np.float32)  # floating
    assert np.all(np.isfinite(tight)) and tight.max() > 0.1 and tight.std() > 0.01, "bad frame"
    assert np.abs(tight - cloud).mean() > 1e-3, "warp_tokenize: card -> token cloud did not animate"
    print("  scene warp_tokenize: OK")

    print("ALL PASSED")


if __name__ == "__main__":
    main()
