"""Smoke test for the public API surface.

Run: `python -m tests.test_public_api` (or under pytest). Verifies that the
engine's advertised entry points are importable and coherent — a guard against
a submodule refactor silently breaking the top-level API — and that a built-in
scene still renders end-to-end through :func:`warp_shaders.render`.
"""

import numpy as np
import warp as wp

import warp_shaders as ws

wp.init()


def test_top_level_symbols():
    for name in ws.__all__:
        assert hasattr(ws, name), f"warp_shaders.{name} missing from package"
    assert isinstance(ws.__version__, str) and ws.__version__


def test_subsystem_namespaces():
    # procedural: noise + sdf reachable
    assert ws.procedural.fbm3 and ws.procedural.simplex3 and ws.procedural.curl3
    assert ws.procedural.sd_sphere and ws.procedural.op_smooth_union
    # engine: shading, atmosphere, volumetrics, post
    assert ws.engine.shade_pbr and ws.engine.shade_material
    assert ws.engine.atmosphere.atmosphere and ws.engine.atmosphere.atmosphere_lut
    assert ws.engine.volumetric.march_clouds and ws.engine.volumetric.hg_phase
    assert ws.engine.post.tonemap and ws.engine.post.bloom and ws.engine.post.godrays
    # textures: 2D/3D/equirect samplers
    assert ws.textures.sample2d and ws.textures.sample3d and ws.textures.sample_equirect
    # every name in each __all__ actually resolves
    for mod in (ws.procedural, ws.engine):
        for name in mod.__all__:
            assert hasattr(mod, name), f"{mod.__name__}.{name} missing"


def test_quality_tiers():
    for tier in ("low", "medium", "high", "ultra"):
        t = ws.get_tier(tier)
        assert t.name == tier and t.raymarch_steps > 0
    ws.set_active("low")
    assert ws.active_tier().name == "low"


def test_host_builders():
    cam = ws.make_camera((0.0, 0.0, 5.0), (0.0, 0.0, 0.0), fov_deg=45.0, aspect=1.6)
    assert cam is not None
    assert cam.aperture == 0.0            # pinhole by default
    assert abs(cam.focus_dist - 5.0) < 1e-4  # focus defaults to eye->target dist
    # depth-of-field camera + its device helpers are exposed
    dof = ws.make_camera((0.0, 0.0, 5.0), (0.0, 0.0, 0.0), aperture=0.1, focus_dist=4.0)
    assert dof.aperture == 0.1 and abs(dof.focus_dist - 4.0) < 1e-4
    assert ws.engine.lens_offset and ws.engine.focus_point
    mat = ws.make_material((0.8, 0.2, 0.2), roughness=0.3, metallic=1.0)
    assert mat is not None
    assert ws.make_light((0.5, 1.0, 0.3), (1.0, 1.0, 1.0), 3.0) is not None


def test_render_roundtrip():
    ws.set_active("low")
    names = [s.name for s in ws.list_scenes()]
    assert "pbr_demo" in names
    img = ws.render("pbr_demo", width=64, height=36, time=0.0)
    assert isinstance(img, np.ndarray)
    assert img.shape[0] == 36 and img.shape[1] == 64
    assert np.all(np.isfinite(img))


def test_atmosphere_luts():
    # transmittance + Hillaire multiple-scattering LUTs bake finite and coherent
    atmo = ws.engine.atmosphere
    tr = atmo.build_transmittance_lut(size=32, device="cpu")
    ms = atmo.build_multiscatter_lut(tr, size=32, device="cpu")
    trn, msn = tr.numpy(), ms.numpy()
    assert trn.shape == (32, 32, 3) and msn.shape == (32, 32, 3)
    assert np.all(np.isfinite(trn)) and np.all(np.isfinite(msn))
    assert np.all(msn >= 0.0)                        # multiscatter is non-negative
    assert atmo.multiscatter_lut and atmo.atmosphere_lut


if __name__ == "__main__":
    test_top_level_symbols()
    print("  top-level symbols:", len(ws.__all__), "OK")
    test_subsystem_namespaces()
    print("  subsystem namespaces (procedural/engine/textures): OK")
    test_quality_tiers()
    print("  quality tiers low..ultra: OK")
    test_host_builders()
    print("  host builders (camera/material/light): OK")
    test_atmosphere_luts()
    print("  atmosphere transmittance + multiscatter LUTs: OK")
    test_render_roundtrip()
    print("  render('pbr_demo') roundtrip: OK")
    print("ALL PASSED")
