# warp-solar-system-shaders

A zero-dependency, **Shadertoy-compatible** WebGL2 playground for writing and
running raymarched GLSL shaders. Drop a `.frag` file into `shaders/`, pick it
from the dropdown, and it renders full-screen with all the standard uniforms
wired up.

The flagship example is a raymarched solar system — a procedurally-textured
planet with a plasma jet, magnetic field rings, orbiting probes, and a
cube-mapped starfield.

## Run it

The player loads shaders with `fetch()`, so it needs to be served over HTTP
(opening `index.html` as a `file://` won't work — browsers block `fetch` there):

```bash
./serve.sh            # → http://localhost:8080/
# or
python3 -m http.server 8080
```

Then open <http://localhost:8080/>.

- **Drag** the canvas to orbit the camera (feeds `iMouse`).
- **Space** pauses, **R** resets time. Same as the on-screen buttons.
- **↻** recompiles the current file — edit a `.frag`, hit reload, see the change.

## Write a shader

Every shader is a Shadertoy-style fragment shader whose entry point is:

```glsl
void mainImage(out vec4 fragColor, in vec2 fragCoord) { ... }
```

The player injects a `#version 300 es` header, the uniforms below, and a `main()`
that calls your `mainImage`. Write only the body — no `#version`, no `out` color,
no `main()`.

### Available uniforms

| Uniform | Type | Meaning |
|---|---|---|
| `iResolution` | `vec3` | viewport size in pixels (`z = 1.0`) |
| `iTime` | `float` | seconds since start |
| `iTimeDelta` | `float` | seconds since previous frame |
| `iFrame` | `int` | frame counter |
| `iMouse` | `vec4` | `xy` = cursor px while dragging; `zw` = click origin (negative when the button is up) |
| `iDate` | `vec4` | `(year, month, day, seconds-in-day)` |

Texture channels (`iChannel0…`), audio, and multi-pass buffers from Shadertoy
are **not** implemented — this is a single-pass image player. Everything the
solar-system shader needs is procedural, so no channels are required.

### Add it to the gallery

1. Save your shader as `shaders/my-shader.frag`.
2. Add an entry to `shaders/manifest.json`:

   ```json
   { "file": "my-shader.frag", "name": "My Shader", "description": "…" }
   ```

3. Reload the page — it appears in the dropdown.

`shaders/template.frag` is a minimal starter to copy.

## Layout

```
index.html            playground UI (canvas + controls + HUD)
js/player.js          WebGL2 runtime + Shadertoy uniform compat
js/app.js             UI wiring: picker, transport, HUD, error display
shaders/
  manifest.json       gallery list
  solar-system.frag   flagship raymarched scene
  template.frag       minimal starter
serve.sh              local HTTP server helper
```

## Notes on portability

Shaders here are plain GLSL ES 3.00 with the Shadertoy uniform convention, so
they paste straight back into <https://shadertoy.com> (drop the `#version`/`out`
wrapper, which Shadertoy adds itself). Keeping to procedural techniques — SDF
raymarching, value noise, fBm — keeps them dependency-free and reusable.
