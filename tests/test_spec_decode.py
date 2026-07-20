"""FM-index draft speculative decoding: exact vs greedy, fewer forwards on repetitive output, correct drafts."""
import numpy as np

from warp_compress.fm_index import FMIndex
from warp_compress.spec_decode import draft, speculative_generate, _greedy


def test_draft_proposes_the_continuation_of_a_recurring_suffix():
    # ...a b c d e ... a b c  -> after the second "a b c", the draft should propose "d e"
    seq = [1, 2, 3, 4, 5, 9, 9, 1, 2, 3]
    fm = FMIndex(np.asarray(seq, np.int64))
    d = draft(fm, seq, max_draft=2)
    assert d == [4, 5]


def test_draft_empty_when_no_prior_match():
    seq = [1, 2, 3, 4, 5]                                 # all-distinct, no suffix recurs
    fm = FMIndex(np.asarray(seq, np.int64))
    assert draft(fm, seq, max_draft=3) == []


class _PeriodicModel:
    """A deterministic 'model' whose greedy next token repeats with a fixed period — so generation is periodic
    and the FM-index draft can predict it. logits shape (1, L, V), argmax at pos i = ids[i+1-period]."""
    def __init__(self, V=20, period=4):
        import torch  # noqa
        self.V, self.period = V, period

    def __call__(self, inp):
        import torch
        ids = inp[0].tolist()
        L = len(ids)
        logits = torch.full((1, L, self.V), -1.0)
        for i in range(L):
            tgt = ids[i + 1 - self.period] if i + 1 - self.period >= 0 else 0
            logits[0, i, tgt] = 1.0
        return type("O", (), {"logits": logits})()


def test_speculative_matches_greedy_exactly():
    m = _PeriodicModel(V=20, period=4)
    prompt = [3, 7, 11, 5]
    g, g_fwd = _greedy(m, prompt, 30)
    s, s_fwd, _ = speculative_generate(m, None, prompt, 30, max_draft=6)
    assert s == g                                        # speculative decoding is exact vs greedy


def test_speculative_uses_fewer_forwards_on_periodic_output():
    m = _PeriodicModel(V=20, period=4)
    prompt = [3, 7, 11, 5]
    _, g_fwd = _greedy(m, prompt, 40)
    _, s_fwd, n = speculative_generate(m, None, prompt, 40, max_draft=8)
    assert g_fwd == 40                                   # greedy = one forward per token
    assert s_fwd < g_fwd                                 # the index draft accepts multiple tokens per forward
    assert n == 40


def test_non_repetitive_is_no_worse_than_greedy():
    m = _PeriodicModel(V=200, period=1)                  # period-1 => constant token, trivially periodic
    prompt = [42]
    s, s_fwd, _ = speculative_generate(m, None, prompt, 20, max_draft=6)
    assert all(t == 42 for t in s) and s_fwd <= 20       # never more forwards than tokens
