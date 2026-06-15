"""Campaign configuration: tiers, sweep grids, RNG seed, and the acceptance gate.

The user selected the **Core** tier with a **deterministic fabrication corner
grid** and **P2 second-stage only if the gate fails**. All numeric knobs for the
whole campaign live here so the drivers stay declarative and the reproduction
harness is deterministic.
"""

from __future__ import annotations

import numpy as np

from inverse_design import config as cfg

# ── Targets and physics anchors ────────────────────────────────────────────
F_D2 = cfg.F_TARGET                      # 87Rb D2 line (Hz), ~384.23 THz
GAMMA_2PI_MHZ = cfg.GAMMA_RB / (2 * np.pi * 1e6)
FIBER_EFFICIENCY = cfg.FIBER_EFFICIENCY

# ── Verification quality for trim / gate / final sims ──────────────────────
# 8 ps ~ 1.5-2 cavity ring-downs at Q~1.3e4: resolves the ~30 GHz linewidth and
# the <=7.6 GHz target far better than the 5 ps optimization default.
TRIM_RUN_TIME = 8e-12
TRIM_GRID_DL = (0.01, 0.01, 0.01)
FINAL_RUN_TIME = 8e-12
# The final sim MUST use the same mesh as the trim: f_res is mesh-sensitive at
# the ~40 GHz level (> the gate tolerance), so a finer final mesh would
# de-calibrate the trim. dl=0.01 is also the established production mesh for the
# whole project (optimization + all four comparison designs). The convergence
# study (CONV_MESH_DL) separately quantifies the residual mesh uncertainty.
FINAL_GRID_DL = (0.01, 0.01, 0.01)

# ── Acceptance gate ────────────────────────────────────────────────────────
GATE_BETA_MIN = 0.80
GATE_ETA_MIN = 0.75
GATE_C_MIN = 20.0
GATE_DF_FLOOR_HZ = 5.0e9                 # absolute floor of the detuning tolerance


def gate_tolerance_hz(kappa_tot_2pi_ghz: float) -> float:
    """|Δf| tolerance = max(5 GHz, 0.25·κ_tot/2π)."""
    return max(GATE_DF_FLOOR_HZ, 0.25 * float(kappa_tot_2pi_ghz) * 1e9)


def check_gate(perf: dict) -> dict:
    """Evaluate the five acceptance criteria on a perf dict.

    Returns {passed: bool, tol_hz, detuning_hz, criteria: {name: bool}, reasons}.
    """
    df_hz = abs(float(perf["f_res_Hz"]) - F_D2)
    tol = gate_tolerance_hz(perf["kappa_tot_2pi_GHz"])
    qd = perf.get("Q_directional", {})
    # waveguide channel dominates: the output face (-x) has the SMALLEST
    # directional Q (largest κ) among the physical loss channels.
    out_q = qd.get(cfg.OUTPUT_FACE)
    others = [v for k, v in qd.items()
              if k not in (cfg.OUTPUT_FACE, "total")]
    wg_dominates = (out_q is not None and others
                    and out_q <= min(others) + 1e-30)
    criteria = {
        "frequency": df_hz <= tol,
        "beta": perf["beta_wg"] >= GATE_BETA_MIN,
        "eta": perf["eta_atom"] >= GATE_ETA_MIN,
        "C": perf["C_atom"] >= GATE_C_MIN,
        "wg_dominates": bool(wg_dominates),
    }
    reasons = []
    if not criteria["frequency"]:
        reasons.append(f"|Δf|={df_hz/1e9:.1f} GHz > tol {tol/1e9:.1f} GHz")
    if not criteria["beta"]:
        reasons.append(f"β={perf['beta_wg']:.3f} < {GATE_BETA_MIN}")
    if not criteria["eta"]:
        reasons.append(f"η={perf['eta_atom']:.3f} < {GATE_ETA_MIN}")
    if not criteria["C"]:
        reasons.append(f"C={perf['C_atom']:.1f} < {GATE_C_MIN}")
    if not criteria["wg_dominates"]:
        reasons.append("waveguide (-x) channel does not dominate loss budget")
    return {
        "passed": all(criteria.values()),
        "tol_hz": tol,
        "detuning_hz": float(perf["f_res_Hz"]) - F_D2,
        "criteria": criteria,
        "reasons": reasons,
    }


# ── Resonance-trim grids ───────────────────────────────────────────────────
# Analytic first guess: s* = f_res/f_D2 ≈ 384.53/384.23 ≈ 1.00078 (f ∝ 1/s).
GLOBAL_BRACKET = (1.0005, 1.0011)        # straddles the analytic guess
# Fig-5 global-scale axis (a few points around the trim region to show the trend).
GLOBAL_FIG_SCALES = [0.9995, 1.0000, 1.0005, 1.0008, 1.0011, 1.0016]
# Fig-5 defect-only axis (applied to the chosen global candidate; 1.000 is the
# candidate itself and is reused, so only the off-center points cost a sim).
DEFECT_FIG_SCALES = [0.997, 1.000, 1.003]
# Output-coupling axis: scale the OUTPUT (left, 4-cell) mirror hole sizes.
# Larger holes -> stronger mirror -> lower κ_wg -> lower β (more confined);
# smaller -> weaker -> higher β (more overcoupled).
OUTPUT_HOLESCALE = [0.90, 1.00, 1.10]
# Contingency only (cell-count rebuild) if the gate fails on β/η.
OUTPUT_CELLCOUNTS = [3, 5]

TRIM_TOL_HZ_DEFAULT = 7.6e9              # nominal target before the candidate's κ is known

# ── Convergence axes (final candidate only; lean Core grid) ────────────────
# FINAL_GRID_DL=0.008, FINAL pml=12, FINAL run_time=8e-12 serve as the shared
# baseline point of each axis (reused, no extra sim).
CONV_MESH_DL = [0.012, 0.010, 0.008, 0.006]   # 0.010 = production (reused)
CONV_PML_LAYERS = [12, 18]               # 24 layers can destabilize the solver
CONV_RUNTIME = [8e-12, 12e-12]           # ring-down / fit-window length (s)

# ── Fabrication tolerance: deterministic corner grid ───────────────────────
TOL_ETCH_NM = 4.0                        # hole-radius etch bias amplitude
TOL_LATTICE_NM = 2.0                     # lattice-constant bias amplitude
TOL_WIDTH_NM = 10.0                      # beam-width bias amplitude
RNG_SEED = 1234


def tolerance_corner_grid() -> list:
    """Deterministic corners + single-axis extremes + nominal (~15 points).

    Each entry: dict(etch_nm, lattice_nm, width_nm) in nm (signed biases).
    """
    e, l, w = TOL_ETCH_NM, TOL_LATTICE_NM, TOL_WIDTH_NM
    pts = [{"etch_nm": 0.0, "lattice_nm": 0.0, "width_nm": 0.0}]   # nominal
    # full 2^3 corners
    for se in (-e, e):
        for sl in (-l, l):
            for sw in (-w, w):
                pts.append({"etch_nm": se, "lattice_nm": sl, "width_nm": sw})
    # single-axis extremes (isolate each parameter's effect)
    for axis, amp in (("etch_nm", e), ("lattice_nm", l), ("width_nm", w)):
        for s in (-amp, amp):
            d = {"etch_nm": 0.0, "lattice_nm": 0.0, "width_nm": 0.0}
            d[axis] = s
            pts.append(d)
    return pts


# ── Atom-position map grid (postprocess of one sim) ────────────────────────
POS_Z_HEIGHTS_NM = np.arange(200, 351, 10).tolist()    # 200..350 nm above surface
POS_Y_OFFSETS_NM = np.arange(-150, 151, 25).tolist()   # lateral offsets (nm)

# ── Mode-isolation scan ────────────────────────────────────────────────────
MODE_SCAN_HALFWINDOW_HZ = 1.0e12         # ±1 THz around D2 (broad)
MODE_SEPARATION_MIN_LINEWIDTHS = 3.0
