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
from warp_shaders.engine.desitter import (
    comoving_event_horizon,
    ds_entropy,
    efolds,
    gibbons_hawking_temperature,
    horizon_radius as ds_horizon_radius,
    hubble_flow_redshift,
    mode_amplitude,
    mode_crossing_time,
    spectral_tilt,
)
from warp_shaders.engine.lensing import (
    einstein_radius,
    fermat_potential,
    fermat_potential_2d,
    image_positions,
    lens_equation,
    magnifications,
    paczynski_magnification,
    time_delay,
)
from warp_shaders.engine.gw import (
    chirp_frequency,
    chirp_mass,
    evolve_peters,
    gw_frequency,
    orbital_frequency,
    peters_merger_time,
    separation_of_time_left,
    strain_amplitude,
)
from warp_shaders.engine.vacuum import (
    allowed_modes,
    casimir_energy,
    casimir_pressure,
    rindler_horizon_distance,
    rindler_worldline,
    schwinger_critical_field,
    schwinger_rate,
    unruh_temperature,
)
from warp_shaders.engine.kerr import (
    bomb_amplitude,
    ergosurface,
    irreducible_mass,
    kerr_entropy,
    kerr_horizons,
    kerr_omega_h,
    kerr_temperature,
    lense_thirring_omega,
    penrose_bound,
    penrose_extract,
    superradiant,
)
from warp_shaders.engine.holoinfo import (
    erasure_correctable,
    five_qubit_stabilizers,
    happy_central_recoverable,
    happy_erased_legs,
    interval_max_flow,
    mera_cut_bonds,
    mera_layers,
)
from warp_shaders.engine.adscft import (
    bh_entropy_evaporating,
    btz_entropy,
    btz_horizon_radius,
    btz_qnm,
    btz_temperature,
    complexity_growth,
    complexity_rate,
    horizon_translation_length,
    lloyd_bound,
    plateau_angle,
    quotient_wall_position,
    scrambling_time,
    thermal_entanglement,
    thermal_interval_entropy,
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

    # ---- BTZ dictionary: quotient geometry, thermal RT / plateau, exact QNMs ----
    L = 1.0
    assert abs(btz_horizon_radius(4.0, L) - 2.0) < 1e-15, "r_h = L sqrt(M)"
    assert abs(btz_temperature(0.3, L) - 0.3 / (2 * np.pi)) < 1e-15, "T = r_h/(2 pi L^2)"
    assert abs(btz_entropy(0.3) - 2 * np.pi * 0.3 / 4) < 1e-15, "S = 2 pi r_h / 4G"
    lam = horizon_translation_length(0.3, L)
    assert abs(lam - 2 * np.pi * 0.3) < 1e-15, "lambda = 2 pi r_h / L"
    xs = [quotient_wall_position(n, lam) for n in range(0, 8)]
    assert xs[0] == 0.0 and all(xs[k] < xs[k + 1] < 1.0 for k in range(7)), \
        "walls must march monotonically toward the fixed point"
    assert quotient_wall_position(-3, lam) == -xs[3], "walls are symmetric about the origin"

    T_th, eps, c_c = 0.35, 0.05, 1.0
    s_small = thermal_interval_entropy(0.01, T_th, eps, c_c)
    assert abs(s_small - (c_c / 3.0) * np.log(0.01 / eps)) < 2e-4, \
        "theta << beta must reduce to the vacuum log (UV blind to T)"
    th_grid = np.linspace(0.5, 2 * np.pi - 0.2, 60)
    ss = [thermal_interval_entropy(float(x), T_th, eps, c_c) for x in th_grid]
    assert all(ss[k] < ss[k + 1] for k in range(59)), "S_th must be monotone in theta"
    slope = (thermal_interval_entropy(40.0, T_th, eps, c_c)
             - thermal_interval_entropy(39.0, T_th, eps, c_c))
    assert abs(slope - np.pi * c_c * T_th / 3.0) < 1e-6, \
        "theta >> beta must be extensive with slope (pi c/3) T"

    S_BH = 1.5
    th_star = plateau_angle(T_th, S_BH, eps, c_c)
    assert np.pi < th_star < 2 * np.pi, "the plateau must exist for these parameters"
    _, w_lo = thermal_entanglement(th_star - 0.01, T_th, S_BH, eps, c_c)
    _, w_hi = thermal_entanglement(th_star + 0.01, T_th, S_BH, eps, c_c)
    assert (not w_lo) and w_hi, "the wrapped saddle must take over exactly at theta*"
    for th in np.linspace(0.3, 2 * np.pi - 0.3, 40):
        sa, wa = thermal_entanglement(float(th), T_th, S_BH, eps, c_c)
        sc, _ = thermal_entanglement(float(2 * np.pi - th), T_th, S_BH, eps, c_c)
        assert abs(sa - sc) <= S_BH + 1e-9, "Araki-Lieb |S(A) - S(A~)| <= S_BH"
        if wa:
            assert abs(abs(sa - sc) - S_BH) < 1e-9, \
                "in the plateau Araki-Lieb must saturate EXACTLY"
    assert plateau_angle(T_th, 1e9, eps, c_c) == 2 * np.pi, \
        "an unreachable plateau must report 2 pi"

    om_re, om_im = btz_qnm(2.0, 0, 0.05, delta=2.0)
    assert om_re == 2.0 and abs(om_im + 4 * np.pi * 0.05) < 1e-15, \
        "fundamental massless QNM: omega = k - 4 pi i T"
    _, im1 = btz_qnm(2.0, 1, 0.05, delta=2.0)
    assert abs((om_im - im1) - 4 * np.pi * 0.05) < 1e-15, "overtones spaced exactly 4 pi T"
    print(f"  BTZ dictionary: OK  (r_h = L sqrt(M); walls -> fixed points; vacuum/extensive "
          f"limits; plateau theta* = {th_star:.3f} with exact Araki-Lieb saturation; "
          f"QNM spacing 4 pi T)")

    # ---- the three BTZ scenes: structural checks ----
    a = _render("btz_quotient", 0.0)     # cold: small lambda, dense walls
    b = _render("btz_quotient", 6.0)     # hot: large lambda, sparse walls
    hh, ww = a.shape[0], a.shape[1]
    cen = np.s_[hh // 2 - 2:hh // 2 + 2, ww // 2 - 2:ww // 2 + 2]
    assert a.mean(axis=2)[cen].mean() > 0.5, "btz_quotient: the horizon axis must glow"
    assert np.abs(a - b).mean() > 5e-3, "btz_quotient: the hole must breathe"
    print(f"  btz_quotient: OK  (horizon axis {a.mean(axis=2)[cen].mean():.3f}, breathes)")

    a = _render("btz_entanglement", 3.0)     # direct phase
    b = _render("btz_entanglement", 7.0)     # plateau: wrapped
    ys, xs2 = np.mgrid[0:hh, 0:ww]
    rad = np.sqrt((xs2 - ww / 2) ** 2 + (ys - hh / 2) ** 2)
    ringm = np.abs(rad - 0.30 * 0.43 * hh) < 2
    ring_a, ring_b = a.mean(axis=2)[ringm].mean(), b.mean(axis=2)[ringm].mean()
    assert ring_b > 1.4 * ring_a, \
        f"btz_entanglement: the horizon must ignite when wrapped ({ring_a:.3f} -> {ring_b:.3f})"
    assert b[..., 2][ringm].mean() > 0.9 * b[..., 0][ringm].mean(), \
        "btz_entanglement: the claimed horizon should turn violet"
    print(f"  btz_entanglement: OK  (horizon ring {ring_a:.3f} -> {ring_b:.3f} across the plateau)")

    a = _render("btz_ringdown", 1.2)     # ringing hard
    b = _render("btz_ringdown", 8.5)     # settled
    bulk = (rad > 0.36 * 0.43 * hh) & (rad < 0.85 * 0.43 * hh)
    # the ripple is spatially structured (a cos^2 quadrupole spiral riding on a smooth
    # base gradient), so damping is measured as the collapse of spatial variance
    std_a = float(a[..., 2][bulk].std())
    std_b = float(b[..., 2][bulk].std())
    assert std_a > 3.0 * std_b, \
        f"btz_ringdown: the ripples must damp out (blue std {std_a:.4f} -> {std_b:.4f})"
    assert np.abs(a - b).mean() > 5e-3, "btz_ringdown: epochs do not differ"
    print(f"  btz_ringdown: OK  (bulk ripple std {std_a:.4f} -> {std_b:.4f}: exact QNM damping)")

    # ---- quantum-information dictionary: the code, the flow, the network ----
    import itertools
    gens = five_qubit_stabilizers()
    assert len(gens) == 4 and all(
        bin(x | z).count("1") == 4 for (x, z) in gens), "[[5,1,3]]: 4 weight-4 generators"
    for g1, g2 in itertools.combinations(gens, 2):
        assert (bin(g1[0] & g2[1]).count("1") + bin(g1[1] & g2[0]).count("1")) % 2 == 0, \
            "stabilizer generators must mutually commute"
    assert erasure_correctable(())
    for k in (1, 2):
        assert all(erasure_correctable(e) for e in itertools.combinations(range(5), k)), \
            f"[[5,1,3]] must correct ANY {k} erasures"
    assert all(not erasure_correctable(e) for e in itertools.combinations(range(5), 3)), \
        "[[5,1,3]] must fail on ANY 3 erasures (no-cloning)"

    flows = []
    for dth in (0.8, 1.6, 2.4, 3.2):
        f, cut = interval_max_flow(dth)
        assert abs(f - cut) < 1e-9, "max flow must equal min cut EXACTLY (MFMC)"
        flows.append((dth, f))
    assert all(flows[k][1] < flows[k + 1][1] for k in range(3)), \
        "flow must grow with the interval"
    devs = [f - 2.0 * np.log(np.sin(0.5 * d)) for (d, f) in flows]
    assert max(devs) - min(devs) < 0.5, \
        f"flow must track the analytic geodesic length 2 ln sin(dth/2) + const ({devs})"

    assert [mera_layers(2 ** k) for k in range(1, 6)] == [1, 2, 3, 4, 5]
    assert all(mera_cut_bonds(2 * l) - mera_cut_bonds(l) == 2 for l in (2, 4, 8, 32)), \
        "doubling the interval must add exactly one severed bond per side"

    for arc in (0.5, 2.0, 3.0, 3.9, 5.5):
        rec, n_er = happy_central_recoverable(arc, 0.30)
        legs = happy_erased_legs(arc, 0.30)
        assert n_er == len(legs)
        assert rec == erasure_correctable(legs), \
            "the geometric wedge rule MUST equal the [[5,1,3]] erasure criterion"
    print(f"  QI dictionary: OK  ([[5,1,3]] exact; MFMC flow==cut, tracks 2 ln sin "
          f"(offsets {['%.2f' % d for d in devs]}); MERA log law; wedge rule == code rule)")

    # ---- the three quantum-information scenes: structural checks ----
    a = _render("holo_bit_threads", 2.0)
    b = _render("holo_bit_threads", 7.0)
    hh2, ww2 = a.shape[0], a.shape[1]
    ys2, xs3 = np.mgrid[0:hh2, 0:ww2]
    rad2 = np.sqrt((xs3 - ww2 / 2) ** 2 + (ys2 - hh2 / 2) ** 2)
    inner = rad2 < 0.80 * 0.43 * hh2
    cyan_a = float((a[..., 2][inner] - a[..., 0][inner]).clip(0).mean())
    cyan_b = float((b[..., 2][inner] - b[..., 0][inner]).clip(0).mean())
    # S(A) is logarithmic, so the honest growth over this sweep is modest (~15-30%)
    assert cyan_b > 1.08 * cyan_a and cyan_a > 0.05, \
        f"holo_bit_threads: the thread bundle must grow with S(A) ({cyan_a:.4f} -> {cyan_b:.4f})"
    print(f"  holo_bit_threads: OK  (thread flux {cyan_a:.4f} -> {cyan_b:.4f})")

    a = _render("holo_mera", 2.0)
    b = _render("holo_mera", 7.0)
    beads_a = int((a.mean(axis=2)[inner] > 0.85).sum())
    beads_b = int((b.mean(axis=2)[inner] > 0.85).sum())
    assert beads_b > beads_a > 0, \
        f"holo_mera: severed-bond beads must multiply as the interval grows ({beads_a} -> {beads_b})"
    assert np.abs(a - b).mean() > 5e-3, "holo_mera: epochs do not differ"
    print(f"  holo_mera: OK  (bead pixels {beads_a} -> {beads_b}: the log law, counted)")

    a = _render("holo_code", 1.0)
    b = _render("holo_code", 8.0)
    cen2 = np.s_[hh2 // 2 - 2:hh2 // 2 + 2, ww2 // 2 - 2:ww2 // 2 + 2]
    star_a = float(a.mean(axis=2)[cen2].mean())
    star_b = float(b.mean(axis=2)[cen2].mean())
    assert star_a > 0.8 and star_b < 0.6, \
        f"holo_code: the central logical qubit must die at the third erased leg ({star_a:.3f} -> {star_b:.3f})"
    rim = np.abs(rad2 - 0.43 * hh2) < 2
    assert float((b[..., 0][rim] - b[..., 2][rim]).mean()) > 0.05, \
        "holo_code: the erased boundary must burn crimson at peak erasure"
    print(f"  holo_code: OK  (logical qubit {star_a:.3f} -> {star_b:.3f}; boundary burnt)")

    # ---- Kerr dictionary: horizons, the third law, the mine, the bomb ----
    import random
    for a_spin in (0.0, 0.3, 0.7, 0.94):
        rp, rm = kerr_horizons(1.0, a_spin)
        assert abs(rp + rm - 2.0) < 1e-12 and abs(rp * rm - a_spin ** 2) < 1e-9, \
            "r_+ + r_- = 2M and r_+ r_- = a^2 must hold exactly"
        assert abs(ergosurface(1.0, a_spin, 0.0) - rp) < 1e-12, "ergosurface touches r_+ at the pole"
        assert abs(ergosurface(1.0, a_spin, np.pi / 2) - 2.0) < 1e-12, "and reaches 2M at the equator"
        rp2, _ = kerr_horizons(1.0, a_spin)
        assert abs(kerr_omega_h(1.0, a_spin) - a_spin / (2.0 * rp2)) < 1e-12
    assert abs(kerr_temperature(1.0, 0.0) - 1.0 / (8 * np.pi)) < 1e-15, "a=0 is Schwarzschild"
    assert kerr_temperature(1.0, 0.9999999) < 1e-3 * kerr_temperature(1.0, 0.0), \
        "T -> 0 at extremality (the third law)"
    assert abs(kerr_entropy(1.0, 0.0) - 4 * np.pi) < 1e-12, "S = 2 pi M r_+ = 4 pi M^2 at a=0"
    assert abs(irreducible_mass(1.0, 1.0) - 1 / np.sqrt(2)) < 1e-12
    assert abs(penrose_bound() - (1 - 2 ** -0.5)) < 1e-15

    mM, aA = 1.0, 0.98
    mirr = irreducible_mass(mM, aA)
    random.seed(7)
    for _ in range(300):
        mM, aA, _ = penrose_extract(mM, aA, random.uniform(0.001, 0.01),
                                    q=random.uniform(0.5, 1.0))
        mi = irreducible_mass(mM, aA)
        assert mi >= mirr - 1e-12, "THE AREA THEOREM: M_irr must never decrease"
        mirr = mi
    mM, aA, tot = 1.0, 0.9999999, 0.0
    for _ in range(30000):
        mM, aA, de = penrose_extract(mM, aA, 5e-5, q=0.9999)
        tot += de
        if aA < 1e-9:
            break
    assert 0.99 * penrose_bound() < tot <= penrose_bound() + 1e-9, \
        f"the near-reversible mine must approach the Penrose bound from below ({tot:.4f})"

    om_H = kerr_omega_h(1.0, 0.9)
    assert superradiant(0.5 * 2 * om_H, 2, om_H) and not superradiant(1.5 * 2 * om_H, 2, om_H) \
        and not superradiant(-0.1, 2, om_H), "amplification iff 0 < omega < m Omega_H"
    assert abs(bomb_amplitude(10, 0.16) - 1.16 ** 10) < 1e-12
    assert abs(lense_thirring_omega(2.0, 1.0, 0.9) / lense_thirring_omega(4.0, 1.0, 0.9)
               - 8.0) < 1e-12, "frame drag falls as 1/r^3"
    print(f"  Kerr dictionary: OK  (horizon identities; third law; area theorem over 300 "
          f"random extractions; mine reaches {tot:.4f} of bound {penrose_bound():.4f}; "
          f"superradiance boundary; 1/r^3 drag)")

    # ---- the three Kerr scenes: structural checks ----
    a = _render("kerr_ergosphere", 0.2)      # chi ~ 0: no ergoregion
    b = _render("kerr_ergosphere", 7.0)      # chi = 0.98: violet shell + drag
    hk, wk = a.shape[0], a.shape[1]
    boxa = a[hk // 2 - 45:hk // 2 + 45, wk // 2 - 55:wk // 2 + 55]
    boxb = b[hk // 2 - 45:hk // 2 + 45, wk // 2 - 55:wk // 2 + 55]
    v_a = float((boxa[..., 2] - boxa[..., 1]).clip(0).mean())
    v_b = float((boxb[..., 2] - boxb[..., 1]).clip(0).mean())
    assert v_b > 4.0 * max(v_a, 1e-4) and v_b > 0.008, \
        f"kerr_ergosphere: the violet ergoregion veil must be born with the spin ({v_a:.4f} -> {v_b:.4f})"
    print(f"  kerr_ergosphere: OK  (ergo veil {v_a:.4f} -> {v_b:.4f} across the spin-up)")

    a = _render("kerr_penrose", 0.5)         # extremal: wide ergo annulus
    b = _render("kerr_penrose", 15.5)        # mined out: annulus pinched shut
    shadow_a = int((a.mean(axis=2)[hk // 2 - 40:hk // 2 + 40, wk // 2 - 40:wk // 2 + 40] < 0.075).sum())
    shadow_b = int((b.mean(axis=2)[hk // 2 - 40:hk // 2 + 40, wk // 2 - 40:wk // 2 + 40] < 0.075).sum())
    assert shadow_b > 1.2 * shadow_a, \
        f"kerr_penrose: the horizon must GROW as the mine runs (area theorem: {shadow_a} -> {shadow_b} px)"
    assert float(a.mean()) > 1.5 * float(b.mean()), \
        "kerr_penrose: the ergoregion glow must die as the spin is mined out"
    print(f"  kerr_penrose: OK  (horizon {shadow_a} -> {shadow_b} px while the annulus dies)")

    a = _render("kerr_superradiance", 2.0)   # first pass
    b = _render("kerr_superradiance", 9.5)   # ratcheted up
    c = _render("kerr_superradiance", 14.0)  # after the burst: quiet
    cy_a = float((a[..., 2] - a[..., 0]).clip(0).mean())
    cy_b = float((b[..., 2] - b[..., 0]).clip(0).mean())
    assert cy_b > 1.5 * cy_a > 0.0, \
        f"kerr_superradiance: the trapped wave must ratchet up (1+g)^n ({cy_a:.4f} -> {cy_b:.4f})"
    assert float(b.mean()) > 1.8 * float(c.mean()), \
        "kerr_superradiance: after the mirror bursts the disk must go quiet"
    print(f"  kerr_superradiance: OK  (wave {cy_a:.4f} -> {cy_b:.4f}, then the bomb goes off)")

    # ---- de Sitter dictionary: the horizon around the observer ----
    H = 0.7
    assert abs(gibbons_hawking_temperature(H) - H / (2 * np.pi)) < 1e-15, "T = H/2pi"
    A_ds = 4 * np.pi * ds_horizon_radius(H) ** 2
    assert abs(ds_entropy(H) - A_ds / 4) < 1e-12, "S = A/4 = pi/H^2"
    ts = [mode_crossing_time(k, H) for k in (1.0, 2.0, 4.0, 8.0)]
    dts = [ts[i + 1] - ts[i] for i in range(3)]
    assert max(dts) - min(dts) < 1e-12 and abs(dts[0] - np.log(2) / H) < 1e-12, \
        "octave-spaced scales must exit the horizon at EQUAL time steps (log law)"
    ns = spectral_tilt(0.008, 0.006)
    ratio = mode_amplitude(4.0, H, ns) / mode_amplitude(1.0, H, ns)
    assert abs(ratio - 4.0 ** (0.5 * (ns - 1))) < 1e-15 and ratio < 1.0, \
        "the tilted spectrum must be an exact red power law"
    assert abs(efolds(1.0, np.e ** 60) - 60) < 1e-9
    assert abs(comoving_event_horizon(H, 3.0) - np.exp(-2.1) / H) < 1e-12, \
        "the comoving horizon must shrink exponentially"
    assert hubble_flow_redshift(0.0, H) == 1.0 and \
        hubble_flow_redshift(0.999 / H, H) > 20.0, "1+z diverges AT the horizon"
    print(f"  de Sitter dictionary: OK  (T=H/2pi; S=A/4; log-spaced crossings "
          f"dln={dts[0]:.4f}; red tilt n_s={ns:.4f}; 60 e-folds; z divergence)")

    # ---- the three de Sitter scenes: structural checks ----
    a = _render("ds_horizon", 1.5)           # sky full of galaxies, wide horizon
    b = _render("ds_horizon", 14.5)          # dark energy ramped: contracted ring, empty sky
    hk, wk = a.shape[0], a.shape[1]
    core_a = a[hk // 2 - 25:hk // 2 + 25, wk // 2 - 30:wk // 2 + 30]
    core_b = b[hk // 2 - 25:hk // 2 + 25, wk // 2 - 30:wk // 2 + 30]
    gal_a = int((core_a.mean(axis=2) > 0.25).sum())
    gal_b = int((core_b.mean(axis=2) > 0.25).sum())
    assert gal_a > gal_b + 3, \
        f"ds_horizon: the sky must EMPTY as the horizon contracts ({gal_a} -> {gal_b} bright px)"
    print(f"  ds_horizon: OK  (bright galaxy pixels {gal_a} -> {gal_b} into the lonely future)")

    a = _render("ds_inflation", 1.0)         # early: modes still oscillating (cyan)
    b = _render("ds_inflation", 6.5)         # late inflation: all frozen (gold)
    gold_a = float((a[..., 0] - a[..., 2]).clip(0).mean())
    gold_b = float((b[..., 0] - b[..., 2]).clip(0).mean())
    assert gold_b > 3.0 * max(gold_a, 1e-4) and gold_b > 0.01, \
        f"ds_inflation: modes must freeze gold as the comoving horizon collapses ({gold_a:.4f} -> {gold_b:.4f})"
    c = _render("ds_inflation", 14.5)        # structure precipitated
    assert float(c.max()) > 0.5, "ds_inflation: proto-galaxies must precipitate after reheating"
    print(f"  ds_inflation: OK  (freeze-out {gold_a:.4f} -> {gold_b:.4f}, then structure)")

    a = _render("ds_thermal", 0.5)           # cold: big ring, many entropy ticks
    b = _render("ds_thermal", 8.0)           # hot: small ring, few ticks
    ring_a = int((a.mean(axis=2) > 0.15).sum())
    ring_b = int((b.mean(axis=2) > 0.15).sum())
    assert ring_a > ring_b, \
        f"ds_thermal: the horizon must SHRINK as Lambda rises (bright px {ring_a} -> {ring_b})"
    print(f"  ds_thermal: OK  (horizon bright px {ring_a} -> {ring_b} as the bath warms)")

    # ---- the quantum vacuum: Unruh, Casimir, Schwinger ----
    acc = 0.8
    assert abs(unruh_temperature(acc) - acc / (2 * np.pi)) < 1e-15, "T = a/2pi"
    # THE TEMPERATURE TRILOGY: kappa/2pi (Hawking), H/2pi (Gibbons-Hawking), a/2pi
    # (Unruh) are the same statement — equal arguments give equal temperatures
    assert abs(unruh_temperature(acc) - gibbons_hawking_temperature(acc)) < 1e-15, \
        "the temperature trilogy must be one identity"
    for tt in (0.0, 0.5, 2.0, 5.0):
        xw, tw = rindler_worldline(acc, tt)
        assert abs(xw * xw - tw * tw - 1 / (acc * acc)) < 1e-9 and xw > abs(tw), \
            "the worldline must ride the hyperbola, never crossing the horizon"
    assert abs(rindler_horizon_distance(2.0) - 0.5) < 1e-15
    assert abs(casimir_pressure(0.5) / casimir_pressure(1.0) - 16.0) < 1e-12, \
        "halve the gap, SIXTEENFOLD the pressure (the 1/d^4 law)"
    hh = 1e-6
    dnum = -(casimir_energy(1.0 + hh) - casimir_energy(1.0 - hh)) / (2 * hh)
    assert abs(dnum - casimir_pressure(1.0)) < 1e-6, "P = -d(E/A)/dd"
    assert allowed_modes(1.0, 10 * np.pi) == 10 and allowed_modes(0.5, 10 * np.pi) == 5
    e_c = schwinger_critical_field()
    sub = schwinger_rate(e_c / 10, e_c) / schwinger_rate(e_c, e_c)
    assert sub < 1e-13, f"below critical the vacuum must NOT break down ({sub:.2e})"
    assert schwinger_rate(2 * e_c, e_c) > schwinger_rate(e_c, e_c), "above: avalanche"
    print(f"  quantum vacuum: OK  (trilogy T=a/2pi==H/2pi; hyperbola identity; Casimir "
          f"16x + P=-dE/dd; Schwinger subcritical ratio {sub:.1e})")

    # ---- the three vacuum scenes: structural checks ----
    a = _render("unruh_horizon", 0.5)        # a ~ 0.25: cold, dark bath
    b = _render("unruh_horizon", 8.0)        # a = 2.5: hot bath at the vertex
    warm_a = float((a[..., 0] - a[..., 2]).clip(0).mean())
    warm_b = float((b[..., 0] - b[..., 2]).clip(0).mean())
    assert warm_b > 2.0 * max(warm_a, 1e-4) and warm_b > 0.004, \
        f"unruh_horizon: the bath must warm with acceleration ({warm_a:.4f} -> {warm_b:.4f})"
    print(f"  unruh_horizon: OK  (bath warmth {warm_a:.4f} -> {warm_b:.4f} as a ramps)")

    a = _render("casimir_plates", 0.5)       # wide gap: weak force, many modes
    b = _render("casimir_plates", 8.0)       # narrow gap: 1/d^4 arrows
    hk, wk = a.shape[0], a.shape[1]
    # amber arrow band at mid-height, outside the plates
    band_a = a[hk // 2 - 4:hk // 2 + 4, :]
    band_b = b[hk // 2 - 4:hk // 2 + 4, :]
    arr_a = float((band_a[..., 0] - band_a[..., 2]).clip(0).mean())
    arr_b = float((band_b[..., 0] - band_b[..., 2]).clip(0).mean())
    assert arr_b > 1.4 * arr_a > 0.0, \
        f"casimir_plates: the force arrows must grow as the gap closes ({arr_a:.4f} -> {arr_b:.4f})"
    print(f"  casimir_plates: OK  (arrow band {arr_a:.4f} -> {arr_b:.4f} into the squeeze)")

    a = _render("schwinger_pairs", 4.0)      # deep sub-critical: NO pairs
    b = _render("schwinger_pairs", 11.9)     # near critical: pairs firing
    c = _render("schwinger_pairs", 12.7)     # breakdown flash
    # magenta positron signature (R and B high, G low) in the gap interior
    core_a = a[hk // 4:3 * hk // 4, wk // 4:3 * wk // 4]
    core_b = b[hk // 4:3 * hk // 4, wk // 4:3 * wk // 4]
    mag_a = float(np.minimum(core_a[..., 0], core_a[..., 2]).clip(0).mean() - core_a[..., 1].mean())
    mag_b = float(np.minimum(core_b[..., 0], core_b[..., 2]).clip(0).mean() - core_b[..., 1].mean())
    assert mag_b > mag_a, \
        f"schwinger_pairs: pairs must appear only near the critical field ({mag_a:.4f} -> {mag_b:.4f})"
    assert float(c.mean()) > 1.6 * float(a.mean()), "schwinger_pairs: the breakdown must flash"
    print(f"  schwinger_pairs: OK  (pair signature {mag_a:.4f} -> {mag_b:.4f}, then breakdown)")

    # ---- gravitational waves: the chirp, exactly ----
    mc = chirp_mass(1.0, 1.0)
    assert abs(chirp_mass(3.0, 1.0) - chirp_mass(1.0, 3.0)) < 1e-15, "M_c symmetric"
    assert abs(gw_frequency(4.0, 2.0) - 2.0 * orbital_frequency(4.0, 2.0)) < 1e-15, \
        "quadrupole: the wave oscillates at TWICE the orbital frequency"
    assert abs(peters_merger_time(1.0, 1, 1) / peters_merger_time(0.5, 1, 1) - 16.0) < 1e-9, \
        "T ~ a^4: halve the separation, SIXTEENFOLD less time left"
    a0 = 1.3
    assert abs(separation_of_time_left(peters_merger_time(a0, 1, 1), 1, 1) - a0) < 1e-12, \
        "a(T(a)) must invert exactly"
    ratio = chirp_frequency(0.5, mc) / chirp_frequency(1.0, mc)
    assert abs(ratio - 2.0 ** 0.375) < 1e-12, \
        f"the chirp must run as (tc-t)^(-3/8) ({ratio} vs {2.0**0.375})"
    # consistency: the chirp law IS Kepler on the Peters trajectory
    t_left = 0.37
    f_a = gw_frequency(separation_of_time_left(t_left, 1, 1), 2.0)
    f_c = chirp_frequency(t_left, mc)
    assert abs(f_a / f_c - 1.0) < 1e-9, "chirp_frequency must equal f_gw(a(t_left))"
    assert strain_amplitude(2.0, mc, 10.0) > strain_amplitude(1.0, mc, 10.0), \
        "the chirp gets LOUDER as it climbs"
    traj = evolve_peters(1.0, 0.6, 1.0, 1.0, 1e-5, 20000)
    e_seq = [e for _, e in traj]
    a_seq = [a for a, _ in traj]
    assert all(e_seq[k + 1] <= e_seq[k] + 1e-15 for k in range(len(e_seq) - 1)), \
        "Peters: eccentricity only ever decreases"
    assert all(a_seq[k + 1] < a_seq[k] for k in range(len(a_seq) - 1)), \
        "Peters: the orbit only ever shrinks"
    assert e_seq[-1] / e_seq[0] < a_seq[-1] / a_seq[0], \
        "circularization: e must die fractionally faster than a over the inspiral"
    print(f"  gravitational waves: OK  (Mc sym; f_gw=2f_orb; T-ratio 16; chirp "
          f"(tc-t)^-3/8; f-consistency; e {e_seq[0]:.2f}->{e_seq[-1]:.3f} vs "
          f"a {a_seq[0]:.2f}->{a_seq[-1]:.3f})")

    # ---- the three GW scenes: structural checks ----
    a = _render("gw_inspiral", 4.0)          # early inspiral: wide, dim wave
    b = _render("gw_inspiral", 12.4)         # endgame: tight, bright spiral
    c = _render("gw_inspiral", 13.6)         # post-merger ringdown
    assert float(b.mean()) > 1.2 * float(a.mean()), \
        f"gw_inspiral: the wave must brighten toward merger ({a.mean():.3f} vs {b.mean():.3f})"
    assert float(c.mean()) > 0.01, "gw_inspiral: ringdown must render"
    print(f"  gw_inspiral: OK  (mean {float(a.mean()):.3f} -> {float(b.mean()):.3f} into merger)")

    a = _render("gw_chirp", 3.0)             # early: low hum revealed
    b = _render("gw_chirp", 13.5)            # full chirp + ringdown revealed
    hk, wk = a.shape[0], a.shape[1]
    right_a = a[:, 2 * wk // 3:]
    right_b = b[:, 2 * wk // 3:]
    assert float(right_b.mean()) > 1.5 * float(right_a.mean()), \
        "gw_chirp: the reveal must sweep left-to-right"
    # the f-track climbs: upper-panel brightness near the top-right appears late
    top_right_b = b[:hk // 4, 3 * wk // 4:]
    top_right_a = a[:hk // 4, 3 * wk // 4:]
    assert float(top_right_b.mean()) > float(top_right_a.mean()) + 0.005, \
        "gw_chirp: the frequency track must scream upward at the end"
    print(f"  gw_chirp: OK  (right-panel {float(right_a.mean()):.4f} -> {float(right_b.mean()):.4f})")

    a = _render("gw_orbits", 1.0)            # eccentric: wide ellipse
    b = _render("gw_orbits", 15.0)           # late: small, round
    # amber eccentricity ledger (R-G contrast in the bar zone, bottom-left)
    zone_a = a[3 * hk // 5:, :wk // 5]
    zone_b = b[3 * hk // 5:, :wk // 5]
    amb_a = float((zone_a[..., 0] - zone_a[..., 2]).clip(0).mean())
    amb_b = float((zone_b[..., 0] - zone_b[..., 2]).clip(0).mean())
    assert amb_b < amb_a, \
        f"gw_orbits: the eccentricity bar must collapse ({amb_a:.4f} -> {amb_b:.4f})"
    print(f"  gw_orbits: OK  (e-ledger {amb_a:.4f} -> {amb_b:.4f} as the waves iron it round)")

    # ---- gravitational lensing: the exact point lens ----
    te = 1.0
    for bb in (0.05, 0.3, 1.0, 2.5):
        tp, tm = image_positions(bb, te)
        assert abs(lens_equation(tp, te) - bb) < 1e-12 and abs(lens_equation(tm, te) - bb) < 1e-12, \
            "both images must satisfy the lens equation exactly"
        assert tp > te and -te < tm < 0, "one image outside the ring, one inside, inverted"
        mp, mm = magnifications(bb, te)
        assert abs(mp + mm - 1.0) < 1e-9, \
            f"the point-lens SIGNED sum rule mu+ + mu- = 1 must hold (got {mp+mm})"
        assert abs(paczynski_magnification(bb) - (abs(mp) + abs(mm))) < 1e-9, \
            "Paczynski A(u) must equal |mu+| + |mu-|"
    assert abs(paczynski_magnification(1.0) - 3.0 / np.sqrt(5.0)) < 1e-12, "A(1) = 3/sqrt(5)"
    assert abs(paczynski_magnification(50.0) - 1.0) < 1e-3, "A -> 1 far from the lens"
    # Fermat: images sit at stationary points of arrival time
    bb = 0.4
    tp, tm = image_positions(bb, te)
    hh = 1e-7
    for t0 in (tp, tm):
        d1 = (fermat_potential(t0 + hh, bb, te) - fermat_potential(t0 - hh, bb, te)) / (2 * hh)
        assert abs(d1) < 1e-6, f"image at {t0} must be a stationary point of arrival time"
    # the inner image is a SADDLE: negative tangential curvature
    ty = 1e-5
    d2t = (fermat_potential_2d(tm, ty, bb, 0.0, te) - 2 * fermat_potential_2d(tm, 0.0, bb, 0.0, te)
           + fermat_potential_2d(tm, -ty, bb, 0.0, te)) / ty ** 2
    assert d2t < 0, "the inner image must be a saddle of the arrival-time surface"
    assert time_delay(bb, te) > 0, "the saddle image arrives LATE"
    assert time_delay(0.8, te) > time_delay(0.2, te), "delay grows with misalignment"
    assert abs(einstein_radius(1.0, 1.0, 2.0) - np.sqrt(2.0)) < 1e-15
    print(f"  gravitational lensing: OK  (lens eq exact; mu+ + mu- = 1; A(1)=3/sqrt5; "
          f"Fermat stationary + saddle curv {d2t:.3f}<0; delay(0.4)={time_delay(0.4, 1.0):.4f})")

    # ---- the three lensing scenes: structural checks ----
    a = _render("lens_arcs", 1.0)            # source far: nearly unlensed
    b = _render("lens_arcs", 7.8)            # near-alignment: Einstein ring
    assert float(b.mean()) > 1.3 * float(a.mean()), \
        f"lens_arcs: the ring must blaze at alignment ({a.mean():.3f} vs {b.mean():.3f})"
    print(f"  lens_arcs: OK  (mean {float(a.mean()):.3f} -> {float(b.mean()):.3f} at the ring)")

    a = _render("lens_microlensing", 1.5)    # far wings: baseline
    b = _render("lens_microlensing", 8.0)    # peak: curve high, images bright
    hk, wk = a.shape[0], a.shape[1]
    sky_a = a[:hk // 2, :]
    sky_b = b[:hk // 2, :]
    assert float(sky_b.mean()) > float(sky_a.mean()), \
        "lens_microlensing: the image pair must brighten at closest approach"
    print(f"  lens_microlensing: OK  (sky {float(sky_a.mean()):.4f} -> {float(sky_b.mean()):.4f} at peak)")

    a = _render("lens_fermat", 0.4)          # nearly aligned: short delay ledger
    b = _render("lens_fermat", 8.0)          # max offset: long delay ledger
    zone_a = a[:, 4 * wk // 5:]
    zone_b = b[:, 4 * wk // 5:]
    amb_a = float((zone_a[..., 0] - zone_a[..., 2]).clip(0).mean())
    amb_b = float((zone_b[..., 0] - zone_b[..., 2]).clip(0).mean())
    assert amb_b > amb_a, \
        f"lens_fermat: the delay ledger must grow with misalignment ({amb_a:.4f} -> {amb_b:.4f})"
    print(f"  lens_fermat: OK  (delay ledger {amb_a:.4f} -> {amb_b:.4f} as the landscape tilts)")

    print("ALL PASSED (29 scenes + LOD sweep + thermodynamics + phase transition "
          "+ RT dictionary + string screening + Page curve + complexity growth "
          "+ BTZ quotient/plateau/ringdown + [[5,1,3]]/MFMC/MERA/HaPPY "
          "+ Kerr horizons/area-theorem/superradiance "
          "+ de Sitter T=H/2pi/S=A/4/log-crossings/red-tilt "
          "+ vacuum trilogy/Casimir-16x/Schwinger-threshold "
          "+ GW chirp -3/8/T~a^4/f_gw=2f_orb/circularization "
          "+ lensing sum-rule/Paczynski/Fermat-stationary)")


if __name__ == "__main__":
    main()
