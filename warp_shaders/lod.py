"""Level-of-detail / quality tiers.

One knob (`low | medium | high | ultra`) scales the sample-count and octave costs
that dominate raymarched procedural rendering, so a scene stays device-agnostic:
the same code runs on CPU (dev/CI) and scales up to high-end GPUs. The default
tier is auto-detected from the active Warp device.

See docs/research/00-foundations.md for the rationale.
"""

from dataclasses import dataclass

import warp as wp


@dataclass(frozen=True)
class QualityTier:
    name: str
    raymarch_steps: int      # max sphere-tracing steps
    shadow_steps: int        # soft-shadow march steps
    ao_steps: int            # ambient-occlusion taps
    noise_octaves: int       # fbm octaves
    volumetric_steps: int    # cloud/smoke march steps
    lut_size: int            # atmosphere/LUT resolution (per axis)
    resolution_scale: float  # render-resolution multiplier (1.0 = full)
    mip_bias: float          # texture LOD bias (+ = blurrier/cheaper)


TIERS = {
    "low":    QualityTier("low",    raymarch_steps=48,  shadow_steps=8,  ao_steps=3,
                          noise_octaves=4, volumetric_steps=24,  lut_size=32,
                          resolution_scale=0.75, mip_bias=1.0),
    "medium": QualityTier("medium", raymarch_steps=96,  shadow_steps=16, ao_steps=5,
                          noise_octaves=5, volumetric_steps=48,  lut_size=64,
                          resolution_scale=1.0,  mip_bias=0.5),
    "high":   QualityTier("high",   raymarch_steps=160, shadow_steps=24, ao_steps=8,
                          noise_octaves=6, volumetric_steps=96,  lut_size=128,
                          resolution_scale=1.0,  mip_bias=0.0),
    "ultra":  QualityTier("ultra",  raymarch_steps=256, shadow_steps=40, ao_steps=12,
                          noise_octaves=8, volumetric_steps=160, lut_size=256,
                          resolution_scale=1.0,  mip_bias=0.0),
}


def get_tier(name: str) -> QualityTier:
    if name not in TIERS:
        raise ValueError(f"unknown quality tier {name!r}; choose from {list(TIERS)}")
    return TIERS[name]


def auto_tier(device: str = "cpu") -> str:
    """Pick a sensible default tier for a device. CPU stays cheap; GPU goes high."""
    dev = str(device)
    if dev.startswith("cuda"):
        try:
            d = wp.get_device(dev)
            gb = getattr(d, "total_memory", 0) / (1024 ** 3)
            return "ultra" if gb >= 24 else "high"
        except Exception:
            return "high"
    return "low"


# Process-wide active tier, set by the CLI (--quality) and read by LOD-aware scenes.
_ACTIVE = TIERS["high"]


def set_active(name: str, device: str = "cpu") -> QualityTier:
    global _ACTIVE
    _ACTIVE = get_tier(auto_tier(device) if name == "auto" else name)
    return _ACTIVE


def active_tier() -> QualityTier:
    return _ACTIVE
