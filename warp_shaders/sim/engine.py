"""Particle simulation engine (NVIDIA Warp) — the physics half of the project.

Unlike the per-pixel shader scenes, this is a stateful particle system that
evolves over time under real forces. A Warp kernel integrates every particle
each step (gravity, thermal buoyancy, drag, cooling); spawning and the
chain-reaction logic live on the host. Particles are splatted additively to an
image with NumPy, giving a volumetric glow.

Particle kinds:
  1 = neutron   (fast spark, short-lived — the chain reaction)
  2 = fireball  (hot, buoyant, expands then rises)
  3 = smoke     (cooled fireball; behaves the same, just colder)
  4 = device    (the bomb casing — pure ballistic fall under gravity)
"""

import numpy as np
import warp as wp


@wp.kernel
def _integrate(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    temp: wp.array(dtype=float),
    age: wp.array(dtype=float),
    life: wp.array(dtype=float),
    kind: wp.array(dtype=wp.int32),
    dt: float,
    g: float,
    buoy: float,
    drag: float,
    cool: float,
):
    i = wp.tid()
    if life[i] <= 0.0 or age[i] >= life[i]:
        return

    v = vel[i]
    p = pos[i]
    t = temp[i]
    k = kind[i]

    if k == 4:
        # Device: ballistic fall under gravity.
        v = v + wp.vec3(0.0, -g, 0.0) * dt
    else:
        # Thermal buoyancy up (~ temperature) fighting gravity, plus drag + cooling.
        v = v + wp.vec3(0.0, buoy * t - g, 0.0) * dt
        v = v * (1.0 - drag * dt)
        t = wp.max(t - cool * dt, 0.0)

    p = p + v * dt

    vel[i] = v
    pos[i] = p
    temp[i] = t
    age[i] = age[i] + dt


# Fireball color ramp: smoke -> red -> orange -> yellow -> white-blue (hottest).
_STOPS = np.array([0.0, 0.15, 0.35, 0.55, 0.80, 1.0])
_RAMP_R = np.array([0.10, 0.45, 0.95, 1.00, 1.00, 0.85])
_RAMP_G = np.array([0.10, 0.10, 0.25, 0.55, 0.95, 0.95])
_RAMP_B = np.array([0.12, 0.10, 0.05, 0.10, 0.55, 1.00])


def splat_points(width, height, pos, col, bright, eye, target,
                 fov_deg=40.0, stamp_radius=3, up=(0.0, 1.0, 0.0)):
    """Additive-splat colored points (per-particle RGB + brightness) to an image.

    Particles with ``bright <= 0`` are skipped (use this to cull back-facing or
    hidden points). Returns an (H, W, 3) float array.
    """
    frame = np.zeros((height, width, 3), np.float32)
    keep = bright > 0.0
    if not keep.any():
        return frame
    pos, col, bright = pos[keep], col[keep], bright[keep]

    eye = np.asarray(eye, np.float32)
    fwd = np.asarray(target, np.float32) - eye
    fwd /= np.linalg.norm(fwd) + 1e-9
    right = np.cross(fwd, np.asarray(up, np.float32))
    right /= np.linalg.norm(right) + 1e-9
    upv = np.cross(right, fwd)

    rel = pos - eye
    cz = rel @ fwd
    front = cz > 0.05
    if not front.any():
        return frame
    rel, col, bright, cz = rel[front], col[front], bright[front], cz[front]
    cx = rel @ right
    cy = rel @ upv

    f = (height * 0.5) / np.tan(np.radians(fov_deg) * 0.5)
    px = np.round(width * 0.5 + (cx / cz) * f).astype(np.int64)
    py = np.round(height * 0.5 - (cy / cz) * f).astype(np.int64)
    contrib = col * bright[:, None]

    sigma2 = 2.0 * (stamp_radius * 0.5 + 0.5) ** 2
    for dy in range(-stamp_radius, stamp_radius + 1):
        for dx in range(-stamp_radius, stamp_radius + 1):
            w = np.exp(-(dx * dx + dy * dy) / sigma2)
            xx = px + dx
            yy = py + dy
            ok = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
            if ok.any():
                np.add.at(frame, (yy[ok], xx[ok]), contrib[ok] * w)
    return frame


class ParticleSystem:
    def __init__(self, max_n: int, device: str = "cpu"):
        self.max_n = max_n
        self.device = device
        self.pos = np.zeros((max_n, 3), np.float32)
        self.vel = np.zeros((max_n, 3), np.float32)
        self.temp = np.zeros(max_n, np.float32)
        self.age = np.zeros(max_n, np.float32)
        self.life = np.zeros(max_n, np.float32)   # 0 => empty slot
        self.kind = np.zeros(max_n, np.int32)
        self.cursor = 0

        self._d_pos = wp.zeros(max_n, dtype=wp.vec3, device=device)
        self._d_vel = wp.zeros(max_n, dtype=wp.vec3, device=device)
        self._d_temp = wp.zeros(max_n, dtype=float, device=device)
        self._d_age = wp.zeros(max_n, dtype=float, device=device)
        self._d_life = wp.zeros(max_n, dtype=float, device=device)
        self._d_kind = wp.zeros(max_n, dtype=wp.int32, device=device)

    def spawn(self, pos, vel, temp, life, kind):
        """Activate ``len(pos)`` particles, wrapping the ring buffer if needed."""
        n = len(pos)
        if n == 0:
            return
        idx = (self.cursor + np.arange(n)) % self.max_n
        self.pos[idx] = pos
        self.vel[idx] = vel
        self.temp[idx] = temp
        self.age[idx] = 0.0
        self.life[idx] = life
        self.kind[idx] = kind
        self.cursor = int((self.cursor + n) % self.max_n)

    def step(self, dt, g, buoy=9.0, drag=1.2, cool=0.35):
        self._d_pos.assign(self.pos)
        self._d_vel.assign(self.vel)
        self._d_temp.assign(self.temp)
        self._d_age.assign(self.age)
        self._d_life.assign(self.life)
        self._d_kind.assign(self.kind)
        wp.launch(_integrate, dim=self.max_n,
                  inputs=[self._d_pos, self._d_vel, self._d_temp, self._d_age,
                          self._d_life, self._d_kind, float(dt), float(g),
                          float(buoy), float(drag), float(cool)],
                  device=self.device)
        wp.synchronize_device(self.device)
        self.pos = self._d_pos.numpy()
        self.vel = self._d_vel.numpy()
        self.temp = self._d_temp.numpy()
        self.age = self._d_age.numpy()

    def alive_mask(self):
        return (self.life > 0.0) & (self.age < self.life)

    # ---------------------------------------------------------------- render
    def render(self, width, height, eye, target, fov_deg=42.0, stamp_radius=4, exposure=1.0):
        """Additive-splat the live particles to an (H, W, 3) image."""
        frame = np.zeros((height, width, 3), np.float32)
        mask = self.alive_mask()
        if not mask.any():
            return frame

        pos = self.pos[mask]
        temp = self.temp[mask]
        kind = self.kind[mask]

        # Camera basis (look-at).
        eye = np.asarray(eye, np.float32)
        fwd = np.asarray(target, np.float32) - eye
        fwd /= np.linalg.norm(fwd) + 1e-9
        right = np.cross(fwd, np.array([0, 1, 0], np.float32))
        right /= np.linalg.norm(right) + 1e-9
        up = np.cross(right, fwd)

        rel = pos - eye
        cz = rel @ fwd
        front = cz > 0.05
        if not front.any():
            return frame
        rel, temp, kind, cz = rel[front], temp[front], kind[front], cz[front]
        cx = rel @ right
        cy = rel @ up

        f = (height * 0.5) / np.tan(np.radians(fov_deg) * 0.5)
        px = width * 0.5 + (cx / cz) * f
        py = height * 0.5 - (cy / cz) * f

        # Per-particle color + brightness.
        t = np.clip(temp, 0.0, 1.0)
        col = np.stack([np.interp(t, _STOPS, _RAMP_R),
                        np.interp(t, _STOPS, _RAMP_G),
                        np.interp(t, _STOPS, _RAMP_B)], axis=1).astype(np.float32)
        bright = (0.15 + 0.9 * t).astype(np.float32)
        # Neutrons: bright cyan sparks. Device: dim grey.
        col[kind == 1] = np.array([0.6, 1.0, 1.0], np.float32)
        bright[kind == 1] = 2.2
        col[kind == 4] = np.array([0.35, 0.35, 0.4], np.float32)
        bright[kind == 4] = 0.4
        # Depth attenuation so nearer particles read brighter.
        bright *= np.clip(60.0 / (cz * cz + 8.0), 0.2, 2.5).astype(np.float32)
        contrib = col * (bright * exposure)[:, None]

        pxi = np.round(px).astype(np.int64)
        pyi = np.round(py).astype(np.int64)
        sigma2 = 2.0 * (stamp_radius * 0.5) ** 2
        for dy in range(-stamp_radius, stamp_radius + 1):
            for dx in range(-stamp_radius, stamp_radius + 1):
                w = np.exp(-(dx * dx + dy * dy) / sigma2)
                xx = pxi + dx
                yy = pyi + dy
                ok = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
                if not ok.any():
                    continue
                np.add.at(frame, (yy[ok], xx[ok]), contrib[ok] * w)
        return frame
