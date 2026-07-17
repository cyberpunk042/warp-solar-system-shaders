"""Process 1 — tokenization: TURN THE GRAPHIC CARD INTO TOKENS.

Operator spec (verbatim): *"TURN THE ELEMENTS EVERY BIT OF THE CARD INTO TOKEN, 100 000 to
1000 000 tokens ... WE CALL IT TOKENIZATION THIS GOES INTO THE LIBS."*

This is a **conserving transform** — it uses what it transforms, it never spawns. The real RTX
board (``gpu_board``) is sampled to a voxel occupancy grid; every occupied bit of the card becomes
a **token**, sub-divided to reach the 100k–1M range the operator asked for. Nothing is added: the
token count is exactly ``(occupied voxels) x sub**3`` — the same matter, resolved finer.

Each token carries a **type id** from the merge codec (``mergecube.compress``): identical pieces of
the board (repeated GDDR7 packages, VRM chokes, the regular PCB) share one id, so the tokens are
coloured **by what they are** — the card read as a stream of tokens, not a picture. That id is the
vocabulary; the cloud is the tokenised card.

Downstream (Process 2) binds these floating tokens into base pairs. This process STOPS at the token
cloud.
"""

from __future__ import annotations

import dataclasses

import numpy as np

import warp_compress.mergecube as mc
from warp_compress.foldcube import _BB, sample_card


@dataclasses.dataclass
class TokenCloud:
    """The card, tokenised. ``positions`` (N,3) float32 world-space token homes; ``colors`` (N,3)
    float32 per-token type colour; ``ids`` (N,) int32 merge-codec type id. Matter is conserved:
    ``n`` equals the occupied-voxel count times ``sub**3`` — no token is fabricated."""

    positions: np.ndarray
    colors: np.ndarray
    ids: np.ndarray
    origin: tuple
    span: tuple

    @property
    def n(self) -> int:
        return int(self.positions.shape[0])


_PHI = 0.6180339887498949  # golden ratio conjugate — even hue spread over the token vocabulary


def _hsv_to_rgb(h, s, v):
    """Vectorised HSV->RGB (h,s,v in [0,1] arrays) -> (N,3) float32."""
    h6 = (h % 1.0) * 6.0
    i = np.floor(h6).astype(np.int32)
    f = h6 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=1).astype(np.float32)


def tokenize_card(sub: int = 2, block: int = 5) -> TokenCloud:
    """Transform every occupied bit of the real board into a token.

    ``sub`` sub-divides each occupied voxel into ``sub**3`` tokens (sub=2 -> ~365k tokens for the
    Blackwell board; sub=3 -> ~1.2M). ``block`` is the merge-codec block edge used only to assign
    each token its type id (colour). Returns a :class:`TokenCloud`.
    """
    occ = sample_card()                      # (nx, ny, nz) uint8, the real card
    nx, ny, nz = occ.shape
    x0, x1, y0, y1, z0, z1 = _BB
    origin = (x0, y0, z0)
    span = (x1 - x0, y1 - y0, z1 - z0)

    # type id per occupied voxel, from the merge codec (identical card pieces share an id)
    unique, index, meta = mc.compress(occ, block=block)
    vox = np.argwhere(occ > 0).astype(np.int64)          # (M,3) occupied voxel coords
    m = vox.shape[0]
    ib, jb, kb = vox[:, 0] // block, vox[:, 1] // block, vox[:, 2] // block
    ids_vox = index[ib, jb, kb].astype(np.int32)         # (M,)

    # sub-divide each occupied voxel into sub**3 tokens (conserving: no new matter, finer sampling)
    s = int(sub)
    off = (np.arange(s) + 0.5) / float(s)                # sub-cell centres in [0,1)
    ox, oy, oz = np.meshgrid(off, off, off, indexing="ij")
    sub_off = np.stack([ox.ravel(), oy.ravel(), oz.ravel()], axis=1)   # (s^3, 3)
    ns = sub_off.shape[0]

    # broadcast voxel base coords + sub-offsets -> (M, s^3, 3) fractional grid coords
    base = vox[:, None, :].astype(np.float32)            # (M,1,3)
    grid = base + sub_off[None, :, :].astype(np.float32) # (M,s^3,3)
    grid = grid.reshape(-1, 3)                           # (N,3)

    scale = np.array([span[0] / nx, span[1] / ny, span[2] / nz], dtype=np.float32)
    orig = np.array(origin, dtype=np.float32)
    positions = (orig[None, :] + grid * scale[None, :]).astype(np.float32)

    ids = np.repeat(ids_vox, ns).astype(np.int32)        # (N,)

    hue = (ids.astype(np.float64) * _PHI) % 1.0
    val = 0.72 + 0.28 * ((ids.astype(np.float64) * 0.113) % 1.0)   # slight per-type brightness spread
    colors = _hsv_to_rgb(hue, np.full_like(hue, 0.62), val)

    return TokenCloud(positions=positions, colors=colors, ids=ids, origin=origin, span=span)
