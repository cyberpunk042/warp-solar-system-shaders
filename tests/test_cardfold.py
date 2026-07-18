"""Tests for the card-fold codec (warp_compress.cardfold) + the fold-and-merge scenes.

The card codec folds a 2-D card in half repeatedly (mirror folds, alternating axes), merging
matching cells, and unfolds exactly. Verifies round-trips, that a self-similar card folds all the
way to a tiny core, and that the warp_card / warp_fold scenes render and animate.

    python -m tests.test_cardfold
"""

import numpy as np

from warp_compress import cardfold as cf


def _mirror_expand(a):
    a = np.hstack([a, a[:, ::-1]])
    return np.vstack([a, a[::-1, :]])


def main():
    # 1. a mirror-symmetric card folds many times down to its small fundamental tile, exactly
    rng = np.random.default_rng(5)
    card = rng.integers(0, 256, (4, 4)).astype(np.int32)
    for _ in range(3):
        card = _mirror_expand(card)                      # 4 -> 8 -> 16 -> 32
    core, levels = cf.fold_levels_card(card, tol=0)
    back = cf.unfold_levels_card(core, levels)
    assert np.array_equal(back, card), "fold/unfold not exact"
    assert len(levels) >= 4, f"self-similar card should fold many times ({len(levels)})"
    assert core.size <= card.size // 8, f"card did not condense enough (core {core.shape})"
    print(f"  fold/unfold exact: OK  ({len(levels)} folds, {card.shape} -> core {core.shape})")

    # 2. full codec round-trips exactly, incl. a non-symmetric card
    cases = {
        "symmetric": card.astype(np.uint8),
        "random": rng.integers(0, 256, (16, 16)).astype(np.uint8),
        "flat": np.full((8, 8), 7, np.uint8),
    }
    for name, grid in cases.items():
        blob = cf.compress(grid, mode="lossless")
        assert np.array_equal(cf.decompress(blob), grid), f"codec round-trip failed: {name}"
    print(f"  card codec: OK  ({len(cases)} cards)")

    # 3. the symmetric card actually compresses
    blob = cf.compress(card.astype(np.uint8), mode="lossless")
    ratio = card.size / len(blob)
    assert ratio > 3.0, f"symmetric card barely compressed (ratio {ratio:.1f})"
    print(f"  card compresses: OK  ({ratio:.1f}x)")

    # 4. lossy stays within tolerance and shrinks
    noisy = (card + rng.integers(-5, 5, card.shape)).clip(0, 255).astype(np.uint8)
    b_ll = cf.compress(noisy, mode="lossless")
    b_ly = cf.compress(noisy, mode="lossy", tol=10)
    back = cf.decompress(b_ly)
    assert back.shape == noisy.shape
    assert int(np.abs(back.astype(int) - noisy.astype(int)).max()) <= 10, "lossy exceeds tol"
    assert len(b_ly) <= len(b_ll), "lossy did not shrink"
    print(f"  lossy dial: OK  (lossless {len(b_ll)} -> lossy {len(b_ly)})")

    # 5. the RTX-board fold scenes render finite and animate (flat board -> cube / X -> unfold)
    import warp as wp
    import warp_shaders as ws
    wp.init()
    for name in ("warp_fold_card",):
        a = np.asarray(ws.render(name, width=110, height=90, time=0.0), np.float32)   # flat board
        b = np.asarray(ws.render(name, width=110, height=90, time=5.0), np.float32)   # folded
        assert np.all(np.isfinite(a)) and a.max() > 0.1 and a.std() > 0.01, f"{name}: bad frame"
        assert np.abs(a - b).mean() > 1e-3, f"{name}: did not animate (fold)"
        print(f"  scene {name}: OK")

    print("ALL PASSED")


if __name__ == "__main__":
    main()
