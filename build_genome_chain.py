"""Build the whole genome ladder as ONE long continuous take.

The eight genome scenes are each already a *process* that re-forms from the previous stage in its opening
seconds then settles. Playing full timelines back-to-back and cross-dissolving every seam turns them into a
single shape retransforming — stage i's settled body dissolves into stage i+1 re-forming from it (which
looks like stage i, so the blend is smooth), then stage i+1 plays its own transformation. Every frame is a
real render of the dedicated scene, so each stage keeps its own beauty; only the seams are blended.

Starts on the real RTX board (stage 0), shows tokenization, ends holding on the merged chromosome.

    python build_genome_chain.py            # -> docs/engine/genome_chain.gif

No cheating: the same matter flows the whole way (each genome library preserves the pair ordering), nothing
is spawned or teleported, and each stage is the physics-checked output of its own library.
"""
from __future__ import annotations

import os
import time as _t

import numpy as np
from PIL import Image

from warp_shaders.scene import get_scene

W, H = 384, 256
FPS = 14
OVERLAP = 11                       # frames cross-dissolved at each seam
DEV = "cpu"
OUT = os.path.join("docs", "engine", "genome_chain.gif")

# (scene, t0, t1, n_frames, mouse) — each genome scene forms from the previous stage over ~0..3.4s
# then settles. Stage 0 is gpu_board — the *exact* board tokenize_card voxel-samples (its board_map is
# the tokenization source) — posed via mouse to the same flat grazing angle as the token field, so the
# dissolve reads as this board becoming its own tokens, not a cut to a different card.
_GB_POSE = (48.0, -88.0)   # gpu_board mouse -> az=0.62, el=0.34 (matches warp_tokenize's camera)
SEGMENTS = [
    ("gpu_board",       0.0, 3.0, 34, _GB_POSE),   # 0 the real source board (posed to the token angle)
    ("warp_tokenize",   0.0, 5.2, 46, (0.0, 0.0)), # 1 that same board -> its tokens
    ("warp_basepair",   0.0, 5.2, 46, (0.0, 0.0)), # 2 tokens -> base pairs
    ("warp_helix",      0.0, 5.6, 48, (0.0, 0.0)), # 3 -> double helices
    ("warp_nucleosome", 0.0, 5.6, 48, (0.0, 0.0)), # 4 -> nucleosome beads
    ("warp_fibre",      0.0, 5.6, 48, (0.0, 0.0)), # 5 -> 30 nm fibre
    ("warp_telomere",   0.0, 5.6, 48, (0.0, 0.0)), # 6 -> telomere-capped strand
    ("warp_chromosome", 0.0, 7.2, 62, (0.0, 0.0)), # 7 -> merged chromosome (held)
]


def _render_segment(name: str, t0: float, t1: float, n: int, mouse=(0.0, 0.0)) -> np.ndarray:
    sc = get_scene(name)
    out = np.empty((n, H, W, 3), np.float32)
    s = _t.time()
    for k in range(n):
        t = t0 + (t1 - t0) * (k / max(n - 1, 1))
        out[k] = np.clip(sc.render(W, H, t, (float(mouse[0]), float(mouse[1])), DEV), 0.0, 1.0)
    print(f"  {name}: {n} frames in {_t.time() - s:.1f}s", flush=True)
    return out


def main() -> None:
    segs = [_render_segment(*seg) for seg in SEGMENTS]

    frames: list[np.ndarray] = []
    carry = None
    for i, sf in enumerate(segs):
        n = sf.shape[0]
        if carry is not None:                       # cross-dissolve prev tail into this head
            for k in range(OVERLAP):
                a = (k + 1) / (OVERLAP + 1)
                frames.append((1.0 - a) * carry[k] + a * sf[k])
        head = 0 if carry is None else OVERLAP
        tail = n if i == len(segs) - 1 else n - OVERLAP
        frames.extend(sf[head:tail])
        carry = None if i == len(segs) - 1 else sf[n - OVERLAP:]

    print(f"total frames {len(frames)}  {len(frames) / FPS:.1f}s", flush=True)

    imgs = [Image.fromarray((f * 255.0 + 0.5).astype(np.uint8), "RGB") for f in frames]
    # one global palette sampled ACROSS every stage (card pastels, grey beads, purple chromosome) so no
    # stage's colours get quantized away.
    n = len(imgs)
    picks = [int(n * fr) for fr in (0.03, 0.12, 0.22, 0.33, 0.45, 0.57, 0.70, 0.82, 0.93, 0.99)]
    montage = Image.fromarray(np.vstack([np.asarray(imgs[min(p, n - 1)]) for p in picks]), "RGB")
    pal = montage.convert("P", palette=Image.ADAPTIVE, colors=200)
    imgs = [im.quantize(palette=pal, dither=Image.FLOYDSTEINBERG) for im in imgs]

    imgs[0].save(OUT, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / FPS), loop=0, optimize=True)
    print(f"WROTE {OUT}  {os.path.getsize(OUT) / 1e6:.1f} MB  {len(imgs)} frames")


if __name__ == "__main__":
    main()
