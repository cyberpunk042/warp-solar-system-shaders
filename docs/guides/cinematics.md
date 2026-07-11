# Cinematics — camera paths, video, and the reel

The engine renders single frames; the cinematics layer turns them into motion:
**keyframed camera moves**, **video encoding** (MP4/WebP/GIF), and a **showcase
reel** that stitches many scenes together. All host-side, no GPU required.

## Video output — `engine.video`

`render.py --video PATH` encodes an animation; the extension picks the container:

```bash
# H.264 MP4 (needs the ffmpeg stack: pip install imageio-ffmpeg)
python render.py --scene solar_system --frames 120 --fps 24 --video out/ss.mp4

# animated WebP / GIF (always available via Pillow, no extra deps)
python render.py --scene aurora --frames 60 --fps 20 --video out/aurora.webp
```

`.mp4` / `.webm` go through **imageio + imageio-ffmpeg** (a self-contained static
ffmpeg). If that stack is missing the writer degrades to an animated `.webp`
beside the requested path and says so — a machine without ffmpeg still produces a
video. `.webp` / `.gif` / `.apng` always use Pillow.

From Python:

```python
from warp_shaders.engine.video import write_video
frames = [scene.render(960, 540, t/24, device="cpu") for t in range(120)]
write_video(frames, "out/clip.mp4", fps=24)     # -> "out/clip.mp4" (or .webp fallback)
```

## Camera paths — `engine.camera_path`

A `CameraPath` is timed **keyframes** (eye, look-at target, vertical FOV). The eye
follows a **Catmull-Rom spline** through the keyframe positions (smooth curved
moves, not robotic corners); target and FOV **ease** between keyframes. Named
easings: `linear`, `smoothstep`, `smoother`, `ease_in`, `ease_out`, `ease_in_out`.

```python
from warp_shaders.engine.camera_path import orbit, dolly, fly

path = orbit(center=(0, 0, 0), radius=6.0, elevation=0.3, turns=1.0)  # circle the subject
path = dolly((0, 0, 8), (0, 0, 3), target=(0, 0, 0), fov0=50, fov1=35) # push in + zoom
path = fly([                                                           # arbitrary keys
    (0.00, (0, 1, 9), (0, 0, 0), 48),
    (0.50, (7, 3, 0), (0, 0, 0), 42),
    (1.00, (0, 1, -9), (0, 0, 0), 48),
], easing="ease_in_out")

eye, target, fov = path.sample(0.4)      # interpolated at 40% along the path
cam = path.camera(0.4, aspect=16/9)      # ...or build the engine Camera directly
```

**Driving a scene with a path.** The cosmos renderer takes a camera override:

```python
from warp_shaders.cosmos import presets
from warp_shaders.cosmos.system import render_system

sys = presets.get("trinary")
path = orbit(radius=sys.dist, elevation=0.3, turns=1.0)
frames = [render_system(sys, 960, 540, time=t, camera=path.sample(t / 12.0 % 1.0))
          for t in [i * 0.1 for i in range(120)]]
```

The built-in `ss_flyby` scene wires exactly this — a looping keyframed sweep around
the trinary system, renderable through the normal `--frames/--video` path.

## The reel — `reel.py`

A **reel** stitches several scenes into one video with crossfade dissolves and an
optional Ken-Burns push-in:

```bash
python reel.py -o out/showcase.mp4 --width 960 --height 540 --fps 24
python reel.py -o out/mini.webp --preset mini --width 480 --height 270
```

A playlist is a list of `Clip`s:

```python
from reel import Clip, render_reel
from warp_shaders.engine.video import write_video

clips = [
    Clip("ss_flyby", seconds=6.0, t0=0.0, t1=6.0, look="cinematic"),
    Clip("earth_v2", seconds=3.5, look="cinematic", zoom=(1.0, 1.12)),  # Ken-Burns in
    Clip("black_hole", seconds=3.5, look="film", fade=0.6),             # 0.6s dissolve out
]
frames = render_reel(clips, 960, 540, fps=24)
write_video(frames, "out/mine.mp4", fps=24)
```

Each `Clip` sets the scene, its duration, the scene-time span to sample (`t0`→`t1`),
a post [look](../api/engine.md#named-looks), a `zoom` start/end (Ken Burns), and the
`fade` seconds joining it to the **next** clip. The `showcase` preset tours the
engine's hero scenes; `mini` is a quick smoke reel.

![cinematic fly-by](../cosmos/ss_flyby.gif)

## Notes

- **Frame rate vs scene time.** `render.py` advances scene `time` by `1/fps` per
  frame, so `--fps` sets both playback speed and the simulation step. For a fixed
  motion at a different playback rate, drive `time` yourself (as the snippets
  above do) and hand the frames to `write_video`.
- **Even dimensions.** H.264 pads to a macro-block multiple; odd sizes grow a few
  pixels. WebP/GIF keep exact dimensions.
- **MP4 dependency.** Only `.mp4`/`.webm` need `imageio-ffmpeg`; everything else
  is pure Pillow.
