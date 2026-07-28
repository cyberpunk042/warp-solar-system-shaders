"""Smoke tests for the AdS/CFT holography set (all six scenes + the dictionary).

Frames are deterministic, so beyond finite/in-range/animating we assert real structure.
ads_cft: boundary ring / tiled bulk / hologram. ads_bulk: horizon shadow, lit CFT sky,
LOD sweep. ads_hawking_page: the two phases. ads_entanglement: the mutual-information
phase transition (geodesic pairings swap; the wedge lights). ads_wormhole: the shadow is
dark with the coupling OFF and fills with the (cyan) dual CFT when ON. ads_confinement:
the string survives any separation below T_HP and breaks on the horizon above it. The
host-side dictionary (entropies, mutual information, turning radius, screening angle) is
asserted against closed forms.

    python -m tests.test_holography
"""

import numpy as np
import warp as wp

import warp_shaders as ws
from warp_shaders import lod
from warp_shaders.engine.adscft import (
    bh_entropy_evaporating,
    complexity_growth,
    complexity_rate,
    lloyd_bound,
    scrambling_time,
    hawking_page_temperature,
    hawking_temperature,
    horizon_radius,
    interval_entropy,
    large_hole_radius,
    mass_of_radius,
    mutual_information,
    page_curve,
    page_time,
    screening_angle,
    string_turning_radius,
)

_DISK_R = 0.43  # keep in sync with warp_shaders/scenes/ads_cft.py


def _render(name, t, w=160, h=120):
    img = np.asarray(ws.render(name, width=w, height=h, time=t), np.float32)
    assert img.shape == (h, w, 3), (name, img.shape)
    assert np.all(np.isfinite(img)), f"{name}: non-finite"
    assert img.min() >= 0.0, f"{name}: negative"
    assert img.max() > 0.1, f"{name}: essentially black"
    assert img.std() > 0.02, f"{name}: flat fill"
    return img


def main():
    wp.init()

    a = _render("ads_cft", 2.0)
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    # same mapping as the kernel: uv in units of height, disk boundary at r = 1
    u = (xx + 0.5 - 0.5 * w) / h
    v = (0.5 * h - (yy + 0.5)) / h
    r = np.sqrt(u * u + v * v) / _DISK_R
    lum = a.mean(axis=2)

    ring = lum[(r > 0.96) & (r < 1.04)].mean()
    bulk = lum[r < 0.85].mean()
    ext = lum[(r > 1.15) & (r < 1.8)].mean()
    # ACES tonemapping compresses highlights, so assert clear margins rather than raw ratios
    assert ring > 1.2 * bulk, f"no conformal-boundary ring (ring {ring:.3f} vs bulk {bulk:.3f})"
    assert ring > 1.2 * ext, f"ring not brighter than exterior ({ring:.3f} vs {ext:.3f})"
    assert lum[r < 0.85].std() > 0.05, "bulk shows no tiling structure"
    assert ext > 0.01, "holographic exterior is empty"
    assert bulk > ext * 0.8, f"bulk unexpectedly dimmer than far exterior ({bulk:.3f} vs {ext:.3f})"
    print(f"  ads_cft t=2.0: OK  (ring {ring:.3f}  bulk {bulk:.3f}  ext {ext:.3f})")

    # the Mobius isometry flow must animate the frame
    b = _render("ads_cft", 6.0)
    assert np.abs(a - b).mean() > 1e-3, "isometry flow did not change the frame"
    print("  ads_cft flow: OK  (frames differ)")

    # ads_bulk — a real horizon shadow against a lit CFT boundary sky
    a = _render("ads_bulk", 3.0)
    lum = a.mean(axis=2)
    # bloom bleeds a little light into the shadow at smoke-test resolution, so the
    # "dark" threshold is 0.04 (empirically the shadow sits at ~0.10 frame fraction)
    dark = float((lum < 0.04).mean())
    assert dark > 0.05, f"ads_bulk: no horizon shadow (dark frac {dark:.3f})"
    assert float((lum > 0.15).mean()) > 0.25, "ads_bulk: boundary CFT sky not lit"
    b = _render("ads_bulk", 6.5)
    assert np.abs(a - b).mean() > 1e-3, "ads_bulk: orbit/flow did not change the frame"
    print(f"  ads_bulk t=3.0: OK  (shadow-frac {dark:.3f}  max {a.max():.3f})")

    # LOD contract: every quality tier renders (bounces 1..4, steps scale with tier)
    prev = lod.active_tier().name
    try:
        for q in ("low", "medium", "high", "ultra"):
            lod.set_active(q)
            _render("ads_bulk", 3.0, w=96, h=72)
        print("  ads_bulk LOD: OK  (low/medium/high/ultra all render)")
    finally:
        lod.set_active(prev)

    # Hawking temperature: T(r_h) = (L² + 3r_h²)/(4πL²r_h) has a MINIMUM at r_h = L/√3
    # (T_min = √3/2πL) — small AdS holes cool as they grow (negative specific heat),
    # large ones heat up (positive specific heat, the Hawking-Page structure).
    l_ads = 7.0
    t_small = hawking_temperature(0.5, l_ads)     # r_h ≈ 0.98, small branch
    t_mid = hawking_temperature(50.0, l_ads)      # r_h ≈ 16, near/above the minimum
    t_large = hawking_temperature(500.0, l_ads)   # r_h ≈ 37, large branch
    t_min = np.sqrt(3.0) / (2.0 * np.pi * l_ads)
    assert t_small > t_min and t_mid > t_min and t_large > t_min, "T below analytic minimum"
    assert t_small > t_mid, f"small branch not cooling with size ({t_small:.4f} vs {t_mid:.4f})"
    assert t_large > t_mid, f"large branch not heating with size ({t_large:.4f} vs {t_mid:.4f})"
    print(f"  hawking_temperature: OK  (T_min {t_min:.4f} < both branches; "
          f"small {t_small:.4f} > mid {t_mid:.4f} < large {t_large:.4f})")

    # Hawking-Page dictionary: T_HP = 1/(πL); the large hole at the transition has
    # r_h = L exactly; mass/radius/temperature round-trip through the engine helpers.
    t_hp = hawking_page_temperature(2.2)
    assert abs(t_hp - 1.0 / (np.pi * 2.2)) < 1e-12
    r_hp = large_hole_radius(t_hp, 2.2)
    assert abs(r_hp - 2.2) < 1e-9, f"large hole at T_HP should sit at r_h = L (got {r_hp:.6f})"
    m_hp = mass_of_radius(r_hp, 2.2)
    assert abs(horizon_radius(m_hp, 2.2) - r_hp) < 1e-6, "mass/radius round-trip broken"
    assert abs(hawking_temperature(m_hp, 2.2) - t_hp) < 1e-9, "T(M(r_h(T_HP))) != T_HP"
    print(f"  hawking_page dictionary: OK  (T_HP {t_hp:.4f}, r_h(T_HP) = L = {r_hp:.4f})")

    # ads_hawking_page — the two phases: hole above T_HP (t=2.0: shadow), thermal AdS
    # below (t=5.9: no shadow, the box glows)
    # (bloom lifts the shadow floor to ~0.05 at smoke resolution, so "dark" is < 0.08 here)
    a = _render("ads_hawking_page", 2.0)
    lum = a.mean(axis=2)
    dark_hole = float((lum < 0.08).mean())
    assert dark_hole > 0.30, f"ads_hawking_page: no nucleated-hole shadow ({dark_hole:.3f})"
    b = _render("ads_hawking_page", 5.9)
    dark_ads = float((b.mean(axis=2) < 0.08).mean())
    assert dark_ads < 0.3 * dark_hole, \
        f"ads_hawking_page: thermal-AdS phase should have (almost) no shadow " \
        f"({dark_ads:.3f} vs hole {dark_hole:.3f})"
    assert np.abs(a - b).mean() > 5e-3, "ads_hawking_page: phases do not differ"
    print(f"  ads_hawking_page: OK  (shadow-frac hole {dark_hole:.3f} vs thermal-AdS {dark_ads:.3f})")

    # ---- RT dictionary: entropy = regularized geodesic length (Calabrese-Cardy) ----
    # S(Δθ) = (c/3) ln((2/ε) sin(Δθ/2)): symmetric under Δθ → 2π − Δθ (pure global
    # state: S_A = S_Ā), maximal at Δθ = π where S = (1/3) ln(2/ε) exactly.
    eps = 0.05
    assert abs(interval_entropy(1.0, eps) - interval_entropy(2.0 * np.pi - 1.0, eps)) < 1e-12
    assert interval_entropy(np.pi, eps) > interval_entropy(1.0, eps) > interval_entropy(0.3, eps)
    assert abs(interval_entropy(np.pi, eps) - np.log(2.0 / eps) / 3.0) < 1e-12
    # mutual information: exactly 0 in the disconnected phase, monotone-on as the gap closes
    i_far, c_far = mutual_information(2.0, 1.35, 1.35, eps)
    i_mid, c_mid = mutual_information(0.5, 1.35, 1.35, eps)
    i_near, c_near = mutual_information(0.2, 1.35, 1.35, eps)
    assert i_far == 0.0 and not c_far, "far intervals should be disconnected with I = 0"
    assert c_mid and c_near and i_near > i_mid > 0.0, "I should switch on and grow as gap closes"
    print(f"  RT dictionary: OK  (S(pi) = ln(2/eps)/3; I: far 0 -> mid {i_mid:.3f} "
          f"-> near {i_near:.3f})")

    # ---- Wilson-string geometry: turning radius + screening angle (closed forms) ----
    r_bdy = 24.0
    assert abs(string_turning_radius(1e-9, r_bdy) - r_bdy) < 1e-5
    assert string_turning_radius(0.99 * np.pi, r_bdy) < 0.2
    r1, r2 = string_turning_radius(1.0, r_bdy), string_turning_radius(2.0, r_bdy)
    assert r1 > r2, "turning radius must deepen with separation"
    th = screening_angle(2.2, r_bdy)
    assert abs(string_turning_radius(th, r_bdy) - 2.2) < 1e-9, "screening-angle inverse broken"
    print(f"  string geometry: OK  (r_min monotone; screening inverse round-trips, "
          f"th_scr(2.2, 24) = {th:.4f})")

    # ---- ads_entanglement: the mutual-information transition on screen ----
    # t=3.3: disconnected (each interval capped by its own geodesic; centre is dim bulk)
    # t=10.5: connected (a near-diameter cross-geodesic passes the centre; wedge glows)
    a = _render("ads_entanglement", 10.5)
    b = _render("ads_entanglement", 3.3)
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    u = (xx + 0.5 - 0.5 * w) / h
    v = (0.5 * h - (yy + 0.5)) / h
    rr = np.sqrt(u * u + v * v) / _DISK_R
    centre = rr < 0.06
    c_conn = a.mean(axis=2)[centre].mean()
    c_disc = b.mean(axis=2)[centre].mean()
    assert c_conn > 2.0 * c_disc, \
        f"connected cross-geodesic should light the centre ({c_conn:.3f} vs {c_disc:.3f})"
    # the wedge wash is purple: in the connected frame the interior right half gains blue
    right_half = (rr < 0.85) & (u > 0.05)
    blue_conn = a[..., 2][right_half].mean()
    blue_disc = b[..., 2][right_half].mean()
    assert blue_conn > 1.2 * blue_disc, \
        f"entanglement wedge should tint the interior ({blue_conn:.3f} vs {blue_disc:.3f})"
    assert np.abs(a - b).mean() > 5e-3, "ads_entanglement: phases do not differ"
    print(f"  ads_entanglement: OK  (centre conn {c_conn:.3f} vs disc {c_disc:.3f}; "
          f"wedge blue {blue_conn:.3f} vs {blue_disc:.3f})")

    # ---- ads_wormhole: the shadow becomes a window when the coupling switches on ----
    a = _render("ads_wormhole", 9.42)          # coupling OFF — non-traversable
    lum = a.mean(axis=2)
    dark_off = float((lum < 0.05).mean())
    assert dark_off > 0.08, f"ads_wormhole: no shadow with coupling off ({dark_off:.3f})"
    b = _render("ads_wormhole", 3.14)          # coupling ON — traversable
    dark_on = float((b.mean(axis=2) < 0.05).mean())
    assert dark_on < 0.35 * dark_off, \
        f"ads_wormhole: throat did not open ({dark_on:.3f} vs {dark_off:.3f})"
    # the light in the former shadow is the OTHER universe's cyan lattice: blue > red there
    shadow = lum < 0.05
    assert b[..., 2][shadow].mean() > 1.3 * b[..., 0][shadow].mean(), \
        "ads_wormhole: traversed light should be the cyan dual CFT"
    print(f"  ads_wormhole: OK  (shadow-frac off {dark_off:.3f} -> on {dark_on:.3f}; "
          f"dual CFT is blue-dominant)")

    # ---- ads_confinement: the string survives thermal AdS, breaks on the horizon ----
    a = _render("ads_confinement", 4.5)        # confined: wide pair, string through centre
    b = _render("ads_confinement", 6.3)        # deconfined: hole + wide pair -> broken
    ch, cw = a.shape[0] // 2, a.shape[1] // 2
    c_conf = a.mean(axis=2)[ch - 8:ch + 8, cw - 8:cw + 8].mean()
    c_brk = b.mean(axis=2)[ch - 8:ch + 8, cw - 8:cw + 8].mean()
    assert c_conf > 3.0 * c_brk, \
        f"confined string should cross the centre, broken must not ({c_conf:.3f} vs {c_brk:.3f})"
    dark_brk = float((b.mean(axis=2) < 0.08).mean())
    assert dark_brk > 0.25, f"ads_confinement: no shadow in the deconfined phase ({dark_brk:.3f})"
    assert np.abs(a - b).mean() > 5e-3, "ads_confinement: phases do not differ"
    print(f"  ads_confinement: OK  (centre confined {c_conf:.3f} vs broken {c_brk:.3f}; "
          f"deconfined shadow-frac {dark_brk:.3f})")

    # ---- Page-curve dictionary: unitarity from saddle competition (closed forms) ----
    T = 16.0
    tp = page_time(T)
    assert abs(tp / T - (1.0 - 2.0 ** -1.5)) < 1e-12
    assert abs(bh_entropy_evaporating(tp, T) - 0.5) < 1e-12, "S_BH(t_page) must be S0/2"
    s0_rad, isl0 = page_curve(0.0, T)
    s_end, _ = page_curve(T, T)
    s_pg, _ = page_curve(tp, T)
    assert s0_rad == 0.0 and not isl0, "radiation entropy must start at 0 (Hawking saddle)"
    assert abs(s_end) < 1e-12, "radiation entropy must return to 0 (unitarity)"
    assert abs(s_pg - 0.5) < 1e-12, "the curve peaks at S0/2 at the Page time"
    _, before = page_curve(tp - 0.01, T)
    _, after = page_curve(tp + 0.01, T)
    assert (not before) and after, "island dominance must switch exactly at t_page"
    ts = np.linspace(0.0, T, 33)
    ss = [page_curve(float(t), T)[0] for t in ts]
    ipk = int(np.argmax(ss))
    assert all(ss[k] <= ss[k + 1] + 1e-12 for k in range(ipk)), "curve must rise to the peak"
    assert all(ss[k] >= ss[k + 1] - 1e-12 for k in range(ipk, 32)), "curve must fall after it"
    print(f"  page_curve dictionary: OK  (t_page/T = {tp / T:.4f}; peak S0/2 at t_page; "
          f"rises then falls; island flips at t_page)")

    # ---- ads_page_curve: dark shadow before t_page, luminous island interior after ----
    a = _render("ads_page_curve", 3.0)         # Hawking phase: information going in
    b = _render("ads_page_curve", 12.0)        # island phase: the interior belongs to the radiation
    ch, cw = a.shape[0] // 2, a.shape[1] // 2
    cen = np.s_[ch - 12:ch + 12, cw - 12:cw + 12]
    lum_h = a.mean(axis=2)[cen].mean()
    lum_i = b.mean(axis=2)[cen].mean()
    assert lum_h < 0.15, f"ads_page_curve: pre-Page shadow should be near-black ({lum_h:.3f})"
    assert lum_i > 3.0 * lum_h, \
        f"ads_page_curve: island phase must light the interior ({lum_i:.3f} vs {lum_h:.3f})"
    assert b[..., 2][cen].mean() > 1.1 * b[..., 0][cen].mean(), \
        "ads_page_curve: the purified radiation should be blue-dominant"
    assert np.abs(a - b).mean() > 5e-3, "ads_page_curve: phases do not differ"
    print(f"  ads_page_curve: OK  (centre pre-Page {lum_h:.3f} -> island {lum_i:.3f}, "
          f"blue-dominant)")

    # ---- complexity dictionary: the Lloyd bound and eternal growth (closed forms) ----
    M, TR = 0.5, 3.0
    bound = lloyd_bound(M)
    assert abs(bound - 2.0 * M / np.pi) < 1e-15
    assert complexity_rate(0.0, M, TR) == 0.0, "growth starts at zero"
    rs = [complexity_rate(float(t), M, TR) for t in np.linspace(0.0, 40.0, 81)]
    assert all(rs[k] < rs[k + 1] for k in range(80)), "rate must rise monotonically"
    assert all(r < bound for r in rs), "the Lloyd bound is never exceeded"
    assert complexity_rate(10.0 * TR, M, TR) > 0.999 * bound, "rate must approach the bound"
    c0, c1, c2 = (complexity_growth(x, M, TR) for x in (0.0, 1.0, 2.0))
    assert c2 - c1 > c1 - c0 > 0.0, "C(t) must be convex at early times"
    late = (complexity_growth(21.0 * TR, M, TR) - complexity_growth(20.0 * TR, M, TR)) / TR
    assert abs(late - bound) < 1e-3 * bound, "late-time growth must be linear at the bound"
    big = complexity_growth(1000.0, M, 1.0)
    assert np.isfinite(big) and abs(big - bound * (1000.0 - np.log(2.0))) < 1e-6, \
        "large-t path must be numerically stable (ln cosh x -> x - ln 2)"
    ts1, ts2 = scrambling_time(100.0, 0.2), scrambling_time(10000.0, 0.2)
    assert abs(ts1 - np.log(100.0) / (2.0 * np.pi * 0.2)) < 1e-12
    assert ts2 == 2.0 * ts1, "t_* = (beta/2pi) ln S: log scrambling"
    print(f"  complexity dictionary: OK  (Lloyd bound 2M/pi = {bound:.4f}; rate rises to "
          f"it, never past; C convex -> linear; t_* log in S)")

    # ---- ads_complexity: boundary freezes, the interior pillar keeps growing ----
    a = _render("ads_complexity", 1.0)
    b = _render("ads_complexity", 12.5)
    ch, cw = a.shape[0], a.shape[1]
    pil = np.s_[int(ch * 0.30):int(ch * 0.42), cw // 2 - 4:cw // 2 + 4]
    flank = np.s_[ch // 2 - 10:ch // 2 + 10, int(cw * 0.36):int(cw * 0.40)]
    pil_a, pil_b = a.mean(axis=2)[pil].mean(), b.mean(axis=2)[pil].mean()
    assert pil_b > 10.0 * max(pil_a, 1e-4), \
        f"ads_complexity: the interior pillar must grow ({pil_a:.4f} -> {pil_b:.4f})"
    assert b.mean(axis=2)[flank].mean() < 0.08, \
        "ads_complexity: the shadow flanks must stay dark around the pillar"
    assert np.abs(a - b).mean() > 5e-3, "ads_complexity: epochs do not differ"
    print(f"  ads_complexity: OK  (pillar {pil_a:.4f} -> {pil_b:.4f}, flanks dark)")

    print("ALL PASSED (8 scenes + LOD sweep + thermodynamics + phase transition "
          "+ RT dictionary + string screening + Page curve + complexity growth)")


if __name__ == "__main__":
    main()
