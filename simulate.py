#!/usr/bin/env python3
"""Run a nuclear / thermonuclear blast simulation and write frames + a report.

Examples
--------
Dropped fission bomb (gravity fall, then detonation), as a GIF::

    python simulate.py --scenario nuclear --drop --gif out/nuke.gif

Thermonuclear (fission primary igniting a fusion secondary)::

    python simulate.py --scenario thermonuclear --drop --gif out/thermo.gif

Print just the chain-reaction report (fast, no image encode)::

    python simulate.py --scenario thermonuclear --no-images
"""

import argparse
import os

import numpy as np
import warp as wp

from warp_shaders.sim import simulate


def tonemap(frame):
    # Filmic-ish: additive HDR -> [0,1]. Reinhard then gamma.
    c = np.clip(frame, 0.0, None)
    c = c / (1.0 + c)
    c = np.power(c, 1.0 / 2.2)
    return (np.clip(c, 0, 1) * 255 + 0.5).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description="Nuclear / thermonuclear blast simulation (Warp).")
    ap.add_argument("--scenario", default="nuclear", choices=["nuclear", "thermonuclear"])
    ap.add_argument("--drop", dest="drop", action="store_true", default=True,
                    help="gravity drop before detonation (default)")
    ap.add_argument("--no-drop", dest="drop", action="store_false",
                    help="detonate in place, no fall")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=270)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--gif", default=None, help="write the run as an animated GIF")
    ap.add_argument("--out-dir", default=None, help="write frame_####.png into this dir")
    ap.add_argument("--final", default=None, help="write only the last frame here")
    ap.add_argument("--no-images", dest="images", action="store_false", default=True,
                    help="skip image encoding; just print the report")
    args = ap.parse_args()

    wp.init()
    frames, report = simulate(args.scenario, drop=args.drop, frames=args.frames,
                              width=args.width, height=args.height,
                              device=args.device, seed=args.seed)

    # --- chain-reaction report ---
    print(f"\n=== {report['scenario'].upper()} blast report ===")
    print(f"dropped under gravity : {report['dropped']}")
    print(f"primary (fission)  peak neutron pop : {report['primary_peak_neutrons']:.3e}"
          f"   energy : {report['primary_energy']:.3f}")
    if report["scenario"] == "thermonuclear":
        print(f"secondary (fusion) peak neutron pop : {report['secondary_peak_neutrons']:.3e}"
              f"   energy : {report['secondary_energy']:.3f}")
    print(f"total energy (arb.) : {report['total_energy']:.3f}")
    gens = report["generations"]
    if gens:
        print("\nframe |  fission n |  fis.E | fusion n |  fus.E   (chain reaction)")
        for row in gens[::max(1, len(gens) // 14)]:
            print(f"{row[0]:5d} | {row[1]:10.2f} | {row[2]:6.3f} | {row[3]:8.2f} | {row[4]:6.3f}")

    if not args.images:
        return

    imgs = [tonemap(fr) for fr in frames]

    if args.final:
        from PIL import Image
        Image.fromarray(imgs[-1], "RGB").save(args.final)
        print(f"\nwrote {args.final}")
    if args.out_dir:
        from PIL import Image
        os.makedirs(args.out_dir, exist_ok=True)
        for k, im in enumerate(imgs):
            Image.fromarray(im, "RGB").save(os.path.join(args.out_dir, f"frame_{k:04d}.png"))
        print(f"wrote {len(imgs)} frames to {args.out_dir}")
    if args.gif:
        from PIL import Image
        os.makedirs(os.path.dirname(os.path.abspath(args.gif)), exist_ok=True)
        pil = [Image.fromarray(im, "RGB") for im in imgs]
        pil[0].save(args.gif, save_all=True, append_images=pil[1:],
                    duration=int(1000 / args.fps), loop=0)
        print(f"\nwrote {args.gif}  ({len(imgs)} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
