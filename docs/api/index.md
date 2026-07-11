# API reference

The public surface, grouped by subsystem. Everything here is reachable from the
package root — `import warp_shaders as ws` — via the namespaces `ws.procedural`,
`ws.engine`, `ws.textures`, `ws.lod`, `ws.post`, and the curated top-level
symbols listed in `warp_shaders.__all__`.

A reminder from [Concepts](../concepts.md): symbols marked **device** are
`@wp.func`/`@wp.kernel` code you call *inside your own kernel*; symbols marked
**host** run in ordinary Python.

| Page | Covers |
|---|---|
| **[procedural](procedural.md)** | noise generators (value / Perlin / simplex / Worley / fBm / ridged / billow / domain-warp / curl, analytic-derivative, tileable), hashes, and the SDF primitive + operator library |
| **[engine](engine.md)** | uniforms (`Camera`/`Light`/`Frame`/`Quality`), `camera_ray_dir`, PBR (`shade_pbr` + terms), `Material`, atmosphere (+ LUT), volumetrics, and the host post pipeline |
| **[textures](textures.md)** | portable 2D / 3D / equirectangular sampling and the host loaders |
| **[lod](lod.md)** | `QualityTier` presets and tier selection |
| **[scene](scene.md)** | the `Scene` contract and the registry (`render`, `list_scenes`, `get_scene`) |

## Top-level symbols

```python
import warp_shaders as ws

ws.__version__                       # str

# namespaces
ws.procedural   ws.engine   ws.textures   ws.lod   ws.post

# scene registry (host)
ws.Scene   ws.render   ws.list_scenes   ws.get_scene

# quality tiers (host)
ws.QualityTier   ws.get_tier   ws.auto_tier   ws.set_active   ws.active_tier

# uniforms + material
ws.Camera   ws.Light   ws.Frame   ws.Quality
ws.make_camera   ws.make_light   ws.make_frame   ws.make_quality
ws.Material   ws.make_material
```
