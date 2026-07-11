"""Environmentally-responsive plants — the "obvious rules" before a mind.

Same L-System engine as :mod:`warp_shaders.scenes.plants`, but the plant now
reacts to its environment through the turtle's tropism layer (ABOP §2.3.4):

- ``phototropism`` — a sapling bends to follow a light that arcs overhead; the
  stem curves toward wherever the light is (``time`` moves the light).
- ``weeping`` — side shoots sag under gravity into a weeping form (gravitropism).
- ``rain_fold`` — as rain sets in over ``time`` the leaves fold shut (nyctinasty).

There is no decision-making here — these are fixed physical responses. A future
"mind" layer chooses *when* to follow the light or close up; the mechanism it
steers is exactly this.

    python render.py --scene phototropism --frames 24 --fps 12 --gif out/photo.gif
    python render.py --scene rain_fold --time 5 -o rain.png
"""

import math
from dataclasses import replace

from ..life import plants as _plants
from ..life.render import render_plant
from ..scene import Scene


def _norm(v):
    m = math.sqrt(sum(c * c for c in v)) or 1.0
    return (v[0] / m, v[1] / m, v[2] / m)


def _frame_cam(lo, hi, time, mouse, extra=1.7):
    cx = float((lo[0] + hi[0]) * 0.5)
    cy = float(lo[1])
    cz = float((lo[2] + hi[2]) * 0.5)
    size = float(max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 1e-3))
    az = 0.5 + time * 0.03 + float(mouse[0]) * 0.01
    dist = size * extra + 2.0
    ty = cy + (hi[1] - lo[1]) * 0.5
    eye = (cx + dist * math.sin(az),
           ty + size * 0.1 + float(mouse[1]) * 0.02,
           cz + dist * math.cos(az))
    return eye, (cx, ty, cz), size, cy


def _phototropism(width, height, time, mouse, device):
    spec = _plants.get_spec("sapling")
    # light arcs from one side of the sky to the other over ~2s; the stem
    # tracks it, so a --frames run sweeps the whole bend
    u = min(max(time / 2.0, 0.0), 1.0)
    ang = (0.15 + 0.7 * u) * math.pi
    lx, ly, lz = math.cos(ang) * 9.0, 5.5, math.sin(ang) * 4.0
    cfg = replace(spec.cfg, light=(lx, ly, lz), light_e=0.14)
    mesh, (lo, hi) = _plants.grow_mesh_env(spec, spec.gens, cfg)
    eye, target, size, gy = _frame_cam(lo, hi, time, mouse, extra=1.9)
    sun = _norm((lx, ly, lz))
    return render_plant(mesh, width, height, eye, target, sun_dir=sun,
                        device=device, fov=42.0, exposure=1.08, ground_y=gy)


def _weeping(width, height, time, mouse, device):
    spec = _plants.get_spec("weeper")
    e = 0.3                                            # gravitropic sag into a weep
    cfg = replace(spec.cfg, tropism=(0.0, -1.0, 0.0), tropism_e=e)
    mesh, (lo, hi) = _plants.grow_mesh_env(spec, spec.gens, cfg)
    eye, target, size, gy = _frame_cam(lo, hi, time, mouse, extra=1.8)
    return render_plant(mesh, width, height, eye, target,
                        sun_dir=(0.5, 0.82, 0.4), device=device,
                        fov=42.0, exposure=1.06, ground_y=gy)


def _rain_fold(width, height, time, mouse, device):
    spec = _plants.get_spec("sapling")
    rain = min(max((time - 1.0) / 5.0, 0.0), 1.0)      # dry -> wet over time
    # a fixed side light, gentle gravity, and leaves folding shut as rain rises
    cfg = replace(spec.cfg, light=(7.0, 6.0, 2.0), light_e=0.05,
                  leaf_fold=rain)
    mesh, (lo, hi) = _plants.grow_mesh_env(spec, spec.gens, cfg)
    eye, target, size, gy = _frame_cam(lo, hi, time, mouse, extra=1.9)
    # sun dims and cools as the rain clouds roll in
    ex = 1.08 - 0.35 * rain
    sun = (0.45, 0.8 - 0.25 * rain, 0.4)
    return render_plant(mesh, width, height, eye, target, sun_dir=sun,
                        device=device, fov=42.0, exposure=ex, ground_y=gy)


def _wind(width, height, time, mouse, device):
    # a gust as a time-varying horizontal tropism: the whole tuft leans downwind
    # and springs back as the gust pulses (still no decision — pure forcing)
    spec = _plants.get_spec("grass")
    gust = 0.05 + 0.05 * math.sin(time * 1.6) + 0.02 * math.sin(time * 4.3)
    wx, wz = math.cos(0.5), math.sin(0.5)
    cfg = replace(spec.cfg, tropism=(wx, -0.15, wz), tropism_e=max(gust, 0.0))
    mesh, (lo, hi) = _plants.grow_mesh_env(spec, spec.gens, cfg)
    eye, target, size, gy = _frame_cam(lo, hi, time, mouse, extra=1.7)
    return render_plant(mesh, width, height, eye, target,
                        sun_dir=(0.5, 0.8, 0.42), device=device,
                        fov=44.0, exposure=1.05, ground_y=gy)


SCENES = [
    Scene(name="phototropism", renderer=_phototropism,
          description="Sapling bending to follow a light arcing overhead "
                      "(phototropism). --time 0..2."),
    Scene(name="weeping", renderer=_weeping,
          description="Weeping plant whose shoots sag under gravity "
                      "(gravitropism)."),
    Scene(name="rain_fold", renderer=_rain_fold,
          description="Leaves folding shut as rain sets in (nyctinasty). "
                      "--time 0..6."),
    Scene(name="wind", renderer=_wind,
          description="Grass tuft swaying as a gust pulses (time-varying "
                      "tropism). --frames 48 --fps 16 for the sway."),
]
