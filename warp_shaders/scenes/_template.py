"""Template scene — copy this to add a new shader.

    cp warp_shaders/scenes/_template.py warp_shaders/scenes/my_scene.py

Then rename the SCENE name, write the kernel, and it's auto-discovered:

    python render.py --list
    python render.py --scene my_scene -o my_scene.png

Underscore-prefixed modules are skipped by the registry, so this file itself
never shows up as a scene.

Porting a Shadertoy shader? The mapping is mechanical:
  - `mainImage(out vec4 c, in vec2 fragCoord)`  -> the kernel body below
  - `iResolution` -> (width, height) args     `iTime` -> time arg
  - `iMouse.xy`   -> mouse arg (pixel coords)
  - swizzles `p.xz = ...` -> rebuild the vec3 component-wise (Warp has no swizzles)
  - `mix` -> wp.lerp   `fract(x)` -> x - wp.floor(x) (or sdf.fract)   `atan(y,x)` -> wp.atan2
Reusable building blocks live in ``warp_shaders/sdf.py``.
"""

import warp as wp

from ..scene import Scene
from ..sdf import fract  # plus hash2d, noise2d, fbm2d, rot2, sd_torus as needed


@wp.kernel
def render_kernel(
    img: wp.array2d(dtype=wp.vec3),
    width: int,
    height: int,
    time: float,
    mouse: wp.vec2,
):
    i, j = wp.tid()  # i = row (0 = top), j = column

    # Centered, aspect-corrected coords (GLSL fragCoord is bottom-up).
    fx = float(j) + 0.5
    fy = float(height - 1 - i) + 0.5
    res = wp.vec2(float(width), float(height))
    uv = wp.vec2((fx - 0.5 * res[0]) / res[1], (fy - 0.5 * res[1]) / res[1])

    col = wp.vec3(
        0.5 + 0.5 * wp.cos(time + uv[0]),
        0.5 + 0.5 * wp.cos(time + uv[1] + 2.0),
        0.5 + 0.5 * wp.cos(time + uv[0] + 4.0),
    )
    d = wp.length(uv)
    col = col * wp.max(0.0, 1.0 - d)

    img[i, j] = col


SCENE = Scene(
    name="template",
    kernel=render_kernel,
    description="Minimal starter — copy _template.py to begin a new scene.",
)
