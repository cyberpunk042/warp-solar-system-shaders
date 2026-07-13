"""Smoke tests for the boards & memory-block scenes.

Renders each at a tiny resolution and asserts finite, in-range, non-degenerate
output — the RAM stick, NVMe SSD, CPU, heatsink, GPU package, graphics card,
motherboard, and the GPU die floorplan assembled from the component parts.

    python -m tests.test_boards
"""

import numpy as np
import warp as wp

import warp_shaders as ws

_SCENES = [
    "ram_stick", "nvme_ssd", "cpu", "heatsink",
    "gpu_package", "graphics_card", "gpu_blower", "gpu_open",
    "gpu_board", "motherboard", "gpu_floorplan",
]


def _check(name, time):
    img = np.asarray(ws.render(name, width=96, height=72, time=time), np.float32)
    assert img.shape == (72, 96, 3), (name, img.shape)
    assert np.all(np.isfinite(img)), f"{name}: non-finite"
    assert img.min() >= 0.0, f"{name}: negative"
    assert img.max() > 0.05, f"{name}: essentially black"
    assert img.std() > 0.01, f"{name}: flat fill"
    print(f"  {name}: OK  (max {img.max():.3f} std {img.std():.3f})")


def main():
    wp.init()
    for n in _SCENES:
        _check(n, 0.4)
    print(f"ALL PASSED ({len(_SCENES)} scenes)")


if __name__ == "__main__":
    main()
