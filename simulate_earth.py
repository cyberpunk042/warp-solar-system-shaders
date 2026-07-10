#!/usr/bin/env python3
"""Simulate Earth under simultaneous global nuclear detonation.

Pick the arsenal and the outcome — one at a time.

Examples
--------
Reality check (planet survives), total re-armed inventory::

    python simulate_earth.py --arsenal total --outcome grounded --gif out/earth_grounded.gif

The honest catastrophe (toxic dead world)::

    python simulate_earth.py --arsenal total --outcome toxic --gif out/earth_toxic.gif

Hypothetical alien 'softron' (planet disperses)::

    python simulate_earth.py --arsenal peak --outcome shatter --gif out/earth_shatter.gif

Just the energy / chain report::

    python simulate_earth.py --arsenal total --no-images
"""

import argparse
import os

import numpy as np
import warp as wp

from warp_shaders.engine import post
from warp_shaders.sim.earth import simulate_earth


def tonemap(frame):
    # Engine post: gentle bloom on the bright detonation flashes, then ACES.
    hdr = post.bloom(np.clip(frame, 0.0, None), threshold=1.6, strength=0.35,
                     radius=max(2, int(frame.shape[1] * 0.01)), passes=2)
    return (post.tonemap(hdr, mode="aces", exposure=1.0) * 255 + 0.5).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description="Earth under global nuclear detonation (Warp).")
    ap.add_argument("--arsenal", default="total", choices=["current", "total", "peak"])
    ap.add_argument("--outcome", default="grounded", choices=["grounded", "toxic", "shatter"])
    ap.add_argument("--frames", type=int, default=140)
    ap.add_argument("--particles", type=int, default=30000)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=400)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--gif", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--final", default=None)
    ap.add_argument("--no-images", dest="images", action="store_false", default=True)
    args = ap.parse_args()

    wp.init()
    frames, rep = simulate_earth(args.arsenal, args.outcome, frames=args.frames,
                                 n=args.particles, width=args.width, height=args.height,
                                 device=args.device, seed=args.seed)

    print(f"\n=== EARTH · arsenal={rep['arsenal']} · outcome={rep['outcome']} ===")
    print(f"warheads               : {rep['warheads']:,}")
    print(f"total yield            : {rep['yield_Mt']:,.0f} Mt  =  {rep['energy_J']:.3e} J")
    print(f"Earth binding energy   : {rep['binding_J']:.3e} J")
    print(f"arsenal / binding      : {rep['ratio_of_binding']:.2e}   (need >= 1 to disperse the planet)")
    print(f"blast dv / escape vel  : {rep['dv_over_vesc']:.2e}   (escape = {rep['vesc_m_s']:,.0f} m/s)")
    print(f"dino-killer / arsenal  : {rep['chicxulub_x_arsenal']:,.0f}x  (and Earth survived that)")
    print(f"\nVERDICT: {rep['verdict']}")

    if not args.images:
        return

    imgs = [tonemap(fr) for fr in frames]
    from PIL import Image
    if args.final:
        Image.fromarray(imgs[-1], "RGB").save(args.final)
        print(f"wrote {args.final}")
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        for k, im in enumerate(imgs):
            Image.fromarray(im, "RGB").save(os.path.join(args.out_dir, f"frame_{k:04d}.png"))
        print(f"wrote {len(imgs)} frames to {args.out_dir}")
    if args.gif:
        os.makedirs(os.path.dirname(os.path.abspath(args.gif)), exist_ok=True)
        pil = [Image.fromarray(im, "RGB") for im in imgs]
        pil[0].save(args.gif, save_all=True, append_images=pil[1:],
                    duration=int(1000 / args.fps), loop=0)
        print(f"wrote {args.gif}  ({len(imgs)} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
