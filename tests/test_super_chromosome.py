"""Recursive super-chromosome: X+Y strands merge into base pairs and refold; lossless + O(depth) access."""
import numpy as np

from warp_compress.super_chromosome import build, pair_strands


def _cluster(k=16, n=300, mut=8, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 4, size=n)
    seqs = []
    for _ in range(k):
        s = base.copy()
        f = rng.integers(0, n, size=mut)
        s[f] = rng.integers(0, 4, size=mut)
        seqs.append(s)
    return seqs


def test_pair_strands_roundtrips_and_dedups():
    x = np.array([0, 1, 2, 3, 0, 1])
    y = np.array([0, 1, 2, 3])                       # shorter -> tail is GAP-padded, sliced away on decode
    ids, book, (lx, ly) = pair_strands(x, y)
    assert (lx, ly) == (6, 4)
    pairs = book[ids]
    assert np.array_equal(pairs[:lx, 0], x)          # X strand recovered exactly
    assert np.array_equal(pairs[:ly, 1], y)          # Y strand recovered exactly
    assert book.shape[0] <= 4 * 4                     # at most V×V distinct base pairs (here far fewer)


def test_super_chromosome_is_lossless():
    seqs = _cluster(seed=1)
    sc = build(seqs)
    dec = sc.decode()
    assert len(dec) == len(seqs)
    assert all(np.array_equal(a, b) for a, b in zip(dec, seqs))


def test_depth_is_log2_and_types_alternate():
    seqs = _cluster(k=16, seed=2)
    sc = build(seqs)
    assert sc.depth == 4                             # ceil(log2(16))
    assert sc.n_leaves == 16
    assert "".join(sc.leaf_kinds) == "XYXYXYXYXYXYXYXY"


def test_fetch_matches_original_at_random_positions():
    seqs = _cluster(k=8, n=256, mut=10, seed=3)
    sc = build(seqs)
    rng = np.random.default_rng(9)
    for _ in range(60):
        li = int(rng.integers(0, len(seqs)))
        p = int(rng.integers(0, len(seqs[li])))
        assert sc.fetch(li, p) == int(seqs[li][p])


def test_non_power_of_two_cluster_still_lossless():
    seqs = _cluster(k=13, n=180, mut=6, seed=4)      # odd counts -> carry-up path
    sc = build(seqs)
    dec = sc.decode()
    assert len(dec) == 13
    assert all(np.array_equal(a, b) for a, b in zip(dec, seqs))


def test_single_chromosome_degenerates_cleanly():
    seqs = _cluster(k=1, n=100, mut=4, seed=5)
    sc = build(seqs)
    assert sc.depth == 0 and sc.n_leaves == 1
    assert np.array_equal(sc.decode()[0], seqs[0])
    assert sc.fetch(0, 50) == int(seqs[0][50])


def test_recursion_beats_raw_at_low_divergence():
    seqs = _cluster(k=16, n=600, mut=6, seed=6)      # ~1% divergence
    sc = build(seqs)
    raw_bits = sum(len(s) for s in seqs) * 2         # 2 bits/token over ACGT
    assert sc.rate()["total_bits"] < raw_bits        # the recursive fold compresses the cluster
