"""A living meadow — seasons, competition for light, and turnover.

Life at the population scale: a patch of L-System plants that live over **years**.
Each plant is born, grows, blooms, senesces and dies; new seedlings appear in the
gaps. The meadow **recolours with the seasons** (green summer → golden autumn →
sparse winter → fresh spring) and the plants **compete for light** — a plant
shaded by taller neighbours grows less and leans toward the open sky.

Deterministic from ``seed`` (a fixed pool of plants with staggered birth/​death
times, so turnover replays identically). Pure NumPy/Python — the scene turns each
plant's `Standing` state into geometry through the usual L-System pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# rough full-grown height per species (for shading + lean)
_SPECIES_H = {"grass": 1.0, "herb": 1.7, "flower": 1.5, "fern": 1.3,
              "bush": 1.9, "sapling": 1.6, "weeper": 1.6, "tree": 4.0}

# leaf palette (bark, green1, green2, flower, dry) — matches turtle._PALETTE order
_SPRING = np.array([[0.42, 0.28, 0.16], [0.22, 0.55, 0.18], [0.40, 0.72, 0.26],
                    [0.90, 0.35, 0.55], [0.70, 0.62, 0.30]], np.float32)
_AUTUMN = np.array([[0.40, 0.26, 0.15], [0.80, 0.45, 0.14], [0.90, 0.62, 0.20],
                    [0.85, 0.30, 0.30], [0.62, 0.42, 0.18]], np.float32)
_WINTER = np.array([[0.34, 0.26, 0.20], [0.45, 0.38, 0.28], [0.52, 0.44, 0.32],
                    [0.55, 0.40, 0.35], [0.48, 0.40, 0.30]], np.float32)


def _smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def season_phase(t: float, year: float = 1.0) -> float:
    """Fraction through the year: 0=spring, .25=summer, .5=autumn, .75=winter."""
    p = (t / year)
    return p - math.floor(p)


def season_palette(phase: float) -> np.ndarray:
    """Leaf palette for the season — greens in spring/summer, gold in autumn,
    muted brown in winter (interpolated smoothly)."""
    if phase < 0.5:                     # spring + summer: green
        return _SPRING.copy()
    if phase < 0.75:                    # autumn: green -> gold
        f = _smooth((phase - 0.5) / 0.25)
        return (_SPRING * (1.0 - f) + _AUTUMN * f).astype(np.float32)
    f = _smooth((phase - 0.75) / 0.25)  # winter: gold -> muted
    return (_AUTUMN * (1.0 - f) + _WINTER * f).astype(np.float32)


def vigor(phase: float) -> float:
    """Canopy vigour 0..1 — full in summer, low in winter (deciduous rhythm)."""
    # peak at summer (~0.3), trough in winter (~0.85)
    return 0.35 + 0.65 * (0.5 + 0.5 * math.cos(2.0 * math.pi * (phase - 0.3)))


@dataclass
class Plant:
    x: float
    z: float
    species: str
    seed: int
    birth: float        # year of birth
    lifespan: float     # years lived


@dataclass
class Standing:
    plant: Plant
    gen: int            # generation to grow to (age × light)
    maturity: float     # 0..1 growth fraction
    height: float       # current height (maturity × species height)
    light: float        # 0..1 available light (1 = full sun)
    lean: tuple         # (x, z) horizontal direction toward open sky (or None)


class Ecosystem:
    def __init__(self, seed: int = 0, pool: int = 46, radius: float = 6.0,
                 species=None, year: float = 1.0, span_years: float = 4.0):
        rng = np.random.default_rng(seed)
        species = species or ["grass", "herb", "flower", "fern", "bush"]
        self.year = year
        self.radius = radius
        self.plants = []
        for i in range(pool):
            ang = rng.uniform(0, 2 * math.pi)
            rad = radius * math.sqrt(rng.uniform(0, 1))
            # ~55% an established starting cohort, the rest recruit over the years
            if i < int(pool * 0.55):
                birth = float(rng.uniform(-0.4, 0.2))
            else:
                birth = float(rng.uniform(0.2, span_years))
            self.plants.append(Plant(
                x=float(rad * math.cos(ang)), z=float(rad * math.sin(ang)),
                species=str(rng.choice(species)), seed=int(rng.integers(1, 9999)),
                birth=birth, lifespan=float(rng.uniform(1.6, 3.2))))

    def _maturity(self, age: float, lifespan: float) -> float:
        if age < 0.0 or age > lifespan:
            return 0.0
        grow = _smooth(age / 0.5)                       # rise over half a year
        senesce = _smooth((lifespan - age) / 0.45)      # fade near end of life
        return grow * senesce

    def standing(self, t: float, compete: bool = True):
        """The alive plants at time `t` with their growth/light/lean state."""
        alive = [p for p in self.plants if p.birth <= t < p.birth + p.lifespan]
        mats, hts = {}, {}
        for p in alive:
            m = self._maturity(t - p.birth, p.lifespan)
            mats[id(p)] = m
            hts[id(p)] = m * _SPECIES_H.get(p.species, 1.5)
        out = []
        for p in alive:
            m = mats[id(p)]
            light, lean = 1.0, None
            if compete:
                shade = 0.0
                vx, vz = 0.0, 0.0
                for q in alive:
                    if q is p:
                        continue
                    dx, dz = p.x - q.x, p.z - q.z
                    d = math.hypot(dx, dz)
                    if d < 2.2 and hts[id(q)] > hts[id(p)] + 0.05:
                        w = (1.0 - d / 2.2)
                        shade += 0.5 * w
                        vx += (dx / (d + 1e-6)) * w      # away from taller neighbour
                        vz += (dz / (d + 1e-6)) * w
                light = float(min(max(1.0 - shade, 0.2), 1.0))
                if vx * vx + vz * vz > 1e-4:
                    n = math.hypot(vx, vz)
                    lean = (vx / n, vz / n)
            spec_gen = _SPECIES_GEN.get(p.species, 6)
            gen = max(1, int(round(spec_gen * m * (0.55 + 0.45 * light))))
            out.append(Standing(p, gen, m, hts[id(p)], light, lean))
        return out


# full-growth generation per species (mirrors PlantSpec.gens)
_SPECIES_GEN = {"grass": 9, "herb": 6, "flower": 6, "fern": 6,
                "bush": 5, "sapling": 7, "weeper": 6, "tree": 7}
