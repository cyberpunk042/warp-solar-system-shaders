"""A neuron firing — an action potential racing down the axon.

The cell body (soma) with branching dendrites (inputs) and one long axon (output);
every ~2 s a bright **action potential** spikes at the soma and propagates down the
axon to the synaptic terminals. Ray-marched SDF. See
``docs/research/24-the-living-body.md``. iMouse orbits.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..engine.uniforms import Camera, camera_ray_dir
from ..subatomic.field import sd_capsule
from ..subatomic.render import orbit_camera
from ..scene import Scene

# --- build the neuron geometry once: capsules (a, b, r) ---
_rng = np.random.RandomState(3)
_caps = []
# dendrites: branches into the upper hemisphere, each splitting once
for k in range(6):
    a = 2.0 * np.pi * k / 6.0
    d = np.array([np.cos(a) * 0.7, 0.5 + 0.4 * _rng.rand(), np.sin(a) * 0.7])
    d = d / np.linalg.norm(d)
    p1 = d * 1.0
    _caps.append((np.zeros(3), p1, 0.09))
    for _ in range(2):
        tip = p1 + (d + _rng.uniform(-0.5, 0.5, 3)) * 0.6
        _caps.append((p1, tip, 0.05))
# axon: down the −y axis with terminal branches
_axon_a = np.array([0.0, -0.4, 0.0], np.float32)
_axon_b = np.array([0.0, -2.6, 0.0], np.float32)
_caps.append((_axon_a, _axon_b, 0.11))
for k in range(3):
    a = 2.0 * np.pi * k / 3.0
    _caps.append((_axon_b, _axon_b + np.array([np.cos(a) * 0.5, -0.4, np.sin(a) * 0.5]), 0.06))
_CA = np.array([c[0] for c in _caps], np.float32)
_CB = np.array([c[1] for c in _caps], np.float32)
_CR = np.array([c[2] for c in _caps], np.float32)


@wp.func
def _map(p: wp.vec3, ca: wp.array(dtype=wp.vec3), cb: wp.array(dtype=wp.vec3),
         cr: wp.array(dtype=float), nc: int) -> float:
    d = wp.length(p) - 0.45                             # soma
    for k in range(nc):
        d = wp.min(d, sd_capsule(p, ca[k], cb[k], cr[k]))
    return d


@wp.func
def _nrm(p: wp.vec3, ca: wp.array(dtype=wp.vec3), cb: wp.array(dtype=wp.vec3),
         cr: wp.array(dtype=float), nc: int) -> wp.vec3:
    e = 0.012
    return wp.normalize(wp.vec3(
        _map(p + wp.vec3(e, 0.0, 0.0), ca, cb, cr, nc) - _map(p - wp.vec3(e, 0.0, 0.0), ca, cb, cr, nc),
        _map(p + wp.vec3(0.0, e, 0.0), ca, cb, cr, nc) - _map(p - wp.vec3(0.0, e, 0.0), ca, cb, cr, nc),
        _map(p + wp.vec3(0.0, 0.0, e), ca, cb, cr, nc) - _map(p - wp.vec3(0.0, 0.0, e), ca, cb, cr, nc)))


@wp.kernel
def neuron_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera,
                  ca: wp.array(dtype=wp.vec3), cb: wp.array(dtype=wp.vec3),
                  cr: wp.array(dtype=float), nc: int, spike_y: float, spike_i: float,
                  width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)
    bg = wp.vec3(0.02, 0.02, 0.04) + wp.vec3(0.03, 0.02, 0.05) * (0.5 + 0.5 * v)

    t = float(0.0)
    hit = int(0)
    for _ in range(110):
        p = ro + rd * t
        d = _map(p, ca, cb, cr, nc)
        if d < 0.002:
            hit = 1
            break
        t = t + d
        if t > 14.0:
            break
    col = bg
    if hit == 1:
        p = ro + rd * t
        n = _nrm(p, ca, cb, cr, nc)
        key = wp.normalize(wp.vec3(0.5, 0.7, 0.6))
        ndl = wp.max(wp.dot(n, key), 0.0)
        fres = wp.pow(1.0 - wp.max(wp.dot(n, -rd), 0.0), 2.5)
        base = wp.vec3(0.75, 0.45, 0.62)               # membrane pink-purple
        col = base * (0.2 + 0.85 * ndl) + base * (fres * 0.5)
        # the action potential: a bright band travelling down the axon (near x,z≈0)
        axialr = wp.length(wp.vec2(p[0], p[2]))
        band = wp.exp(-((p[1] - spike_y) * (p[1] - spike_y)) / 0.03) * wp.exp(-axialr * axialr * 3.0)
        col = col + wp.vec3(1.0, 0.95, 0.5) * (band * spike_i * 3.0)
        # soma flash at spike onset
        somad = wp.length(p)
        col = col + wp.vec3(1.0, 0.9, 0.6) * (wp.exp(-(somad / 0.5) * (somad / 0.5) * 2.0) * spike_i * 0.8)
    img[i, j] = col


def _render(width, height, time, mouse, device, period=2.4):
    prog = (time % period) / period
    # spike launches at the soma (y≈0) and travels to the axon tip (y≈−2.6)
    spike_y = 0.2 - 3.0 * wp.clamp(prog / 0.55, 0.0, 1.0)
    spike_i = 0.0
    if prog < 0.6:
        spike_i = 1.0
    cam = orbit_camera(width, height, time, mouse, dist=6.5, fov=44.0, el0=0.12,
                       auto=0.15)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(neuron_kernel, dim=(height, width),
              inputs=[img, cam, wp.array(_CA, dtype=wp.vec3, device=device),
                      wp.array(_CB, dtype=wp.vec3, device=device),
                      wp.array(_CR, dtype=wp.float32, device=device), int(len(_CR)),
                      float(spike_y), float(spike_i), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(2, int(min(width, height) * 0.012))
    hdr = post.bloom(hdr, threshold=1.2, strength=0.4, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="neuron",
    description="A neuron firing — soma + branching dendrites + a long axon, with a "
                "bright action potential spiking at the soma and racing down the "
                "axon to the synapses. iMouse orbits; --frames animates the spike.",
    renderer=_render,
)
