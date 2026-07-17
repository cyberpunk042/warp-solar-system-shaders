"""Process 2 — base-pair bounding: bind the floating tokens into base pairs.

Operator spec (verbatim): *"we use the card that is no more a card but a bunch of tokens floating in
the air, and we use those to FORM PAIRS, BASE PAIRS ... for 100000 this mean at least 50000 base
pairs."*

A separate conserving process. Its INPUT is the token cloud from Process 1 (the card, tokenised and
floating). Its output binds those tokens **in twos** — every token joins exactly one pair, so N tokens
become N/2 base pairs (365744 tokens -> 182872 pairs). Nothing is spawned and nothing is destroyed:
the pairs are made *of* the tokens that already exist.

Pairing is by spatial adjacency (tokens that float near each other bind — physically plausible), and
each token is tagged with a DNA base (A/C/G/T from its merge-codec type) so a pair reads as a base
pair, coloured by its two bases. This process stops at the field of base pairs.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .tokenize import tokenize_card

# DNA base palette (A/T/G/C) — the two tokens of a pair are coloured by their base
_BASES = np.array([
    [0.95, 0.35, 0.38],   # A — red
    [0.98, 0.80, 0.30],   # T — gold  (A-T pair)
    [0.35, 0.62, 0.98],   # G — blue
    [0.40, 0.90, 0.65],   # C — green (G-C pair)
], dtype=np.float32)


# the ordered base-pair field (Process 2's OUTPUT) — a wide, shallow lattice of vertical rungs. The
# scene binds the cloud onto this; Process 3 (the double helix) winds it from here. Shared so the two
# processes use the exact same positions (a real chain, not a restart).
_FIELD_NX = 432
_FIELD_SX = 0.115
_FIELD_SZ = 0.150
_FIELD_Y = 1.15
_HL = 0.13                        # rung half-length (token A below the site, token B above)


def ordered_field_sites(n: int) -> np.ndarray:
    """The (jittered) lattice sites the base pairs settle onto — the centre of each rung."""
    gi = np.arange(n)
    nz = int(np.ceil(n / _FIELD_NX))
    frac = lambda v: v - np.floor(v)
    jx = (frac(np.sin(gi * 12.9898) * 43758.5453) - 0.5) * (0.7 * _FIELD_SX)
    jz = (frac(np.sin(gi * 78.2330 + 2.0) * 43758.5453) - 0.5) * (0.7 * _FIELD_SZ)
    jy = (frac(np.sin(gi * 37.7190 + 4.0) * 43758.5453) - 0.5) * 0.10
    x = (gi % _FIELD_NX - _FIELD_NX * 0.5) * _FIELD_SX + jx
    z = (gi // _FIELD_NX - nz * 0.5) * _FIELD_SZ + jz
    y = _FIELD_Y + jy
    return np.stack([x, y, z], axis=1).astype(np.float32)


@dataclasses.dataclass
class BasePairs:
    """The floating tokens, bound in twos. Arrays are (P,·) with P = N/2 pairs. ``a_*``/``b_*`` are the
    two member tokens in the floating cloud; ``field_a``/``field_b`` are the same two tokens in the
    ordered base-pair field (Process 2's OUTPUT — the unwound ladder); ``mid`` the binding centre,
    ``axis`` the unit A->B rung direction. Conserved: every one of the N input tokens appears in exactly
    one pair — none spawned, none dropped."""

    a_pos: np.ndarray
    b_pos: np.ndarray
    a_col: np.ndarray
    b_col: np.ndarray
    mid: np.ndarray
    axis: np.ndarray
    field_a: np.ndarray = None
    field_b: np.ndarray = None

    @property
    def n_pairs(self) -> int:
        return int(self.a_pos.shape[0])


def _disperse_cloud(positions, disperse=1.0):
    """Replicate the Process-1 dispersion (numpy) — the token homes as they float, the INPUT to
    pairing. Matches ``scenes/warp_tokenize._token_pos`` at the given disperse, turb=0."""
    n = positions.shape[0]
    t = np.arange(n, dtype=np.float64)
    frac = lambda v: v - np.floor(v)
    r1 = frac(np.sin(t * 12.9898 + 0.5) * 43758.5453)
    r2 = frac(np.sin(t * 78.2330 + 1.3) * 43758.5453)
    r3 = frac(np.sin(t * 37.7190 + 2.7) * 43758.5453)
    ang = r1 * 6.2831853
    out = np.stack([np.cos(ang), np.zeros_like(ang), np.sin(ang)], axis=1)
    spread = (0.8 + 1.5 * r2)[:, None]
    rise = (0.4 + 1.7 * r3)[:, None]
    disp = out * (disperse * spread) + np.array([0.0, 1.0, 0.0]) * (disperse * rise)
    return (positions + disp).astype(np.float32)


def _morton_order(p):
    """Sort index that walks space locally (Morton / Z-order on 8-bit-quantised coords), so that
    consecutive tokens are spatial neighbours — good partners to bind."""
    lo = p.min(0)
    hi = p.max(0)
    q = np.clip(((p - lo) / np.maximum(hi - lo, 1e-6) * 255.0), 0, 255).astype(np.uint32)

    def _spread(v):                      # interleave 8 bits with two zero gaps (3D Morton)
        v = (v | (v << 8)) & 0x00F00F
        v = (v | (v << 4)) & 0x0C30C3
        v = (v | (v << 2)) & 0x249249
        return v

    code = _spread(q[:, 0]) | (_spread(q[:, 1]) << 1) | (_spread(q[:, 2]) << 2)
    return np.argsort(code, kind="stable")


def bind_pairs(sub: int = 2, block: int = 5, disperse: float = 1.0) -> BasePairs:
    """Bind the floating token cloud into base pairs. Returns a :class:`BasePairs` (N/2 pairs)."""
    tc = tokenize_card(sub=sub, block=block)
    homes = _disperse_cloud(tc.positions, disperse=disperse)     # the floating cloud (Process-1 output)
    ids = tc.ids

    n = homes.shape[0]
    if n % 2:                                                    # keep it conserved: even token count
        homes, ids = homes[:-1], ids[:-1]
        n -= 1

    order = _morton_order(homes)
    a_idx = order[0::2]
    b_idx = order[1::2]

    a_pos = homes[a_idx]
    b_pos = homes[b_idx]
    a_base = (ids[a_idx] & 3)
    b_base = (a_base ^ 1)                                        # complement within the pair (A-T, G-C)
    a_col = _BASES[a_base]
    b_col = _BASES[b_base]

    mid = 0.5 * (a_pos + b_pos)
    d = b_pos - a_pos
    axis = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)

    # the ordered base-pair field (Process 2's output): each pair's two tokens on a vertical rung
    sites = ordered_field_sites(n // 2)
    up = np.array([0.0, _HL, 0.0], np.float32)
    field_a = (sites - up).astype(np.float32)
    field_b = (sites + up).astype(np.float32)

    return BasePairs(a_pos=a_pos, b_pos=b_pos, a_col=a_col, b_col=b_col,
                     mid=mid, axis=axis.astype(np.float32), field_a=field_a, field_b=field_b)
