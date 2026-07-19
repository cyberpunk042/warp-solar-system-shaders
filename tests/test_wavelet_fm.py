"""Wavelet matrix (access/rank/select) and FM-index (count/locate) over token sequences."""
import numpy as np

from warp_compress.fm_index import FMIndex, suffix_array
from warp_compress.wavelet import WaveletMatrix


def _seq(n=6000, V=64, seed=1):
    rng = np.random.default_rng(seed)
    p = 1.0 / np.arange(1, V + 1)
    p /= p.sum()
    return rng.choice(V, size=n, p=p), V


def test_wavelet_access_reconstructs():
    seq, _ = _seq()
    wm = WaveletMatrix(seq)
    assert all(wm.access(i) == seq[i] for i in range(len(seq)))


def test_wavelet_rank_matches_naive():
    seq, V = _seq()
    wm = WaveletMatrix(seq)
    for c in range(0, V, 7):
        for i in (0, 1, len(seq) // 3, len(seq)):
            assert wm.rank(c, i) == int(np.count_nonzero(seq[:i] == c))


def test_wavelet_select_matches_naive():
    seq, V = _seq()
    wm = WaveletMatrix(seq)
    for c in range(0, V, 5):
        occ = np.flatnonzero(seq == c)
        for k in range(0, len(occ), max(1, len(occ) // 4)):
            assert wm.select(c, k) == int(occ[k])
        assert wm.select(c, len(occ)) == -1                       # out of range


def test_suffix_array_is_sorted():
    seq, _ = _seq(n=1500)
    s = np.concatenate([seq + 1, [0]])
    sa = suffix_array(s)
    for a, b in zip(sa[:-1], sa[1:]):
        assert tuple(s[a:]) <= tuple(s[b:])                       # suffixes in lex order


def test_fm_count_and_locate_match_naive():
    seq, _ = _seq(n=4000)
    fm = FMIndex(seq, sa_sample=16)
    rng = np.random.default_rng(5)
    for _ in range(10):
        at = int(rng.integers(0, len(seq) - 3))
        pat = seq[at:at + 3]
        naive = [i for i in range(len(seq) - len(pat) + 1) if np.array_equal(seq[i:i + len(pat)], pat)]
        assert fm.count(pat) == len(naive)
        assert fm.locate(pat) == naive


def test_fm_absent_pattern_is_zero():
    seq, V = _seq(n=2000)
    fm = FMIndex(seq)
    assert fm.count([V + 5, V + 6]) == 0                          # symbols outside the alphabet


def test_fm_predict_next_beats_uniform_on_markov():
    import math
    rng = np.random.default_rng(1)
    V = 16
    trans = rng.dirichlet(np.ones(V) * 0.25, size=(V, V))
    seq = np.empty(40000, np.int64)
    seq[:2] = [0, 1]
    for i in range(2, len(seq)):
        seq[i] = rng.choice(V, p=trans[seq[i - 2], seq[i - 1]])
    fm = FMIndex(seq[:35000], sa_sample=32)
    ll = n = 0.0
    for i in rng.integers(35002, len(seq), 500):
        d = fm.predict_next([int(seq[i - 2]), int(seq[i - 1])])
        ll += -math.log2(d.get(int(seq[i]), 1e-6)); n += 1
    assert ll / n < math.log2(V) * 0.75          # clearly better than uniform (the index recovers structure)


def test_fm_prob_next_context_beats_unigram():
    import math
    rng = np.random.default_rng(2)
    V = 16
    trans = rng.dirichlet(np.ones(V) * 0.25, size=(V, V))
    seq = np.empty(30000, np.int64)
    seq[:2] = [0, 1]
    for i in range(2, len(seq)):
        seq[i] = rng.choice(V, p=trans[seq[i - 2], seq[i - 1]])
    fm = FMIndex(seq[:26000], sa_sample=32)
    uni = ctx = m = 0.0
    for i in rng.integers(26002, len(seq), 400):
        c = int(seq[i])
        uni += -math.log2(max(fm.prob_next([], c, max_order=0), 1e-12))
        ctx += -math.log2(max(fm.prob_next([int(seq[i - 2]), int(seq[i - 1])], c, max_order=4), 1e-12))
        m += 1
    assert ctx / m < uni / m - 0.5          # variable-order context clearly beats the unigram
