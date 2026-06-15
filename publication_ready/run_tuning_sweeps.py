"""Resonance trim onto the Rb D2 line + Fig-5 tuning axes.

Primary job (P0): bring invDes(seed) from 384.53 THz onto 384.23 THz within
the gate tolerance via a global in-plane scale, measured by a 2-point bracket
and linear interpolation (f ∝ 1/s is only approximate because thickness/width
are fixed). Also samples a defect-only scale axis around the chosen candidate
for Fig. 5.

Outputs:
  data/tuning_global_scale.csv/json   (global-scale axis incl. invDes(seed) @ s=1)
  data/tuning_defect_scale.csv/json   (defect-only axis around the candidate)
  data/final_candidate.json           (selected geometry + gate verdict)

    python -m publication_ready.run_tuning_sweeps --dry-run   # local validation
    python -m publication_ready.run_tuning_sweeps             # cloud trim
"""

from __future__ import annotations

import argparse

import numpy as np

from inverse_design import config as cfg
from inverse_design import objective

from publication_ready import campaign, paths
from publication_ready.lib import datasets, layout_ops, reuse

F_D2 = cfg.F_TARGET


def _row(knob, value, perf, marker="", selected=False, hdf5=""):
    det = (perf["f_res_Hz"] - F_D2) / 1e9
    return {
        "knob": knob, "value": value,
        "f_res_THz": perf["f_res_Hz"] / 1e12, "detuning_GHz": det,
        "Q": perf["Q"], "kappa_tot_2pi_GHz": perf["kappa_tot_2pi_GHz"],
        "beta_wg": perf["beta_wg"], "C_atom": perf["C_atom"],
        "eta_atom": perf["eta_atom"], "Vmode_lambda_n3": perf["Vmode_lambda_n3"],
        "g_2pi_GHz_at_250nm": perf["g_2pi_Hz_at_250nm"] / 1e9,
        "design_marker": marker, "is_selected": selected, "hdf5": hdf5,
    }


def dry_run():
    """Validate every candidate geometry locally (no cloud)."""
    layout, context, par = layout_ops.load_layout_from_record(paths.FINAL_RECORD_DIR)
    span0 = layout["positions"][-1] - layout["positions"][0]
    print(f"[dry] loaded invDes(seed): {par.n_holes} holes, span {span0:.3f} um")
    scales = sorted(set(list(campaign.GLOBAL_BRACKET) + [1.0005, 1.0008]))
    for s in scales:
        L = layout_ops.scale_layout(layout, s)
        probs = objective.validate_layout(par, L)
        span = L["positions"][-1] - L["positions"][0]
        print(f"[dry] global s={s:.6f}: span {span:.4f} um, "
              f"valid={not probs} {probs if probs else ''}")
    Lsel = layout_ops.scale_layout(layout, 1.0008)
    for s_d in campaign.DEFECT_FIG_SCALES:
        L = layout_ops.scale_defect_region(Lsel, par, s_d)
        probs = objective.validate_layout(par, L)
        print(f"[dry] defect s_d={s_d:.4f}: valid={not probs} "
              f"{probs if probs else ''}")
    print("[dry] all candidate geometries built and validated locally.")


def global_trim(layout, par):
    """Bracket -> interpolate -> verify -> (optional) refine. Returns points."""
    points = []   # (scale, perf, save_name)

    # invDes(seed) baseline point at s=1 (reuse existing sim; no new credit)
    perf0, _ = reuse.perf_for_label("invDes(seed)")
    points.append((1.0, perf0, paths.HDF5_REGISTRY["invDes(seed)"].name))
    print(f"[trim] s=1.000000 (invDes(seed)): f={perf0['f_res_Hz']/1e12:.4f} THz "
          f"det={(perf0['f_res_Hz']-F_D2)/1e9:+.1f} GHz")

    def run(s):
        L = layout_ops.scale_layout(layout, s)
        name = f"global_s{s:.6f}"
        perf, _, _ = reuse.verify_or_reuse(L, par, name, campaign.TRIM_RUN_TIME,
                                           campaign.TRIM_GRID_DL)
        det = (perf["f_res_Hz"] - F_D2) / 1e9
        print(f"[trim] s={s:.6f}: f={perf['f_res_Hz']/1e12:.4f} THz "
              f"det={det:+.1f} GHz  beta={perf['beta_wg']:.3f}  "
              f"C={perf['C_atom']:.1f}  eta={perf['eta_atom']:.3f}")
        points.append((s, perf, name))
        return perf

    sa, sb = campaign.GLOBAL_BRACKET
    pa, pb = run(sa), run(sb)
    fa, fb = pa["f_res_Hz"], pb["f_res_Hz"]
    m = (fb - fa) / (sb - sa)                       # df/ds (Hz per unit s)
    s_star = sa + (F_D2 - fa) / m
    s_star = float(np.clip(s_star, 0.999, 1.002))
    print(f"[trim] slope df/ds = {m:.3e} Hz; interpolated s* = {s_star:.6f}")
    p_star = run(s_star)

    gate = campaign.check_gate(p_star)
    if not gate["criteria"]["frequency"]:
        # one Newton refinement from the two scales nearest the target
        ranked = sorted(points[1:], key=lambda t: abs(t[1]["f_res_Hz"] - F_D2))
        (s1, q1, _), (s2, q2, _) = ranked[0], ranked[1]
        m2 = (q2["f_res_Hz"] - q1["f_res_Hz"]) / (s2 - s1)
        s_ref = float(np.clip(s1 + (F_D2 - q1["f_res_Hz"]) / m2, 0.999, 1.002))
        print(f"[trim] refine -> s = {s_ref:.6f}")
        run(s_ref)
    return points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        dry_run()
        return

    paths.ensure_dirs()
    layout, context, par = layout_ops.load_layout_from_record(paths.FINAL_RECORD_DIR)

    # ── global trim ────────────────────────────────────────────────────────
    points = global_trim(layout, par)
    # selected = closest to D2 among the simulated (non-baseline) candidates
    cand = [p for p in points if p[0] != 1.0]
    s_sel, perf_sel, name_sel = min(cand, key=lambda t: abs(t[1]["f_res_Hz"] - F_D2))
    gate = campaign.check_gate(perf_sel)
    print(f"\n[trim] SELECTED global s={s_sel:.6f}: "
          f"det={(perf_sel['f_res_Hz']-F_D2)/1e9:+.2f} GHz  "
          f"gate_passed={gate['passed']}  {gate['reasons']}")

    g_rows = []
    for s, perf, name in sorted(points, key=lambda t: t[0]):
        marker = ("invDes(seed)" if s == 1.0
                  else ("final" if abs(s - s_sel) < 1e-12 else ""))
        g_rows.append(_row("global_scale", s, perf, marker,
                           selected=abs(s - s_sel) < 1e-12, hdf5=name))
    datasets.write_csv(paths.DATA_DIR / "tuning_global_scale.csv", g_rows,
                       datasets.TUNING_COLUMNS)
    datasets.write_json(paths.DATA_DIR / "tuning_global_scale.json",
                        {"rows": g_rows, "selected_scale": s_sel,
                         "gate": gate}, with_convention=True)

    # ── defect-only axis around the selected candidate (Fig 5) ──────────────
    L_sel = layout_ops.scale_layout(layout, s_sel)
    d_rows = []
    for s_d in campaign.DEFECT_FIG_SCALES:
        if abs(s_d - 1.0) < 1e-12:
            perf, name = perf_sel, name_sel
        else:
            L = layout_ops.scale_defect_region(L_sel, par, s_d)
            name = f"defect_sd{s_d:.4f}"
            perf, _, _ = reuse.verify_or_reuse(L, par, name, campaign.TRIM_RUN_TIME,
                                               campaign.TRIM_GRID_DL)
            print(f"[defect] s_d={s_d:.4f}: det="
                  f"{(perf['f_res_Hz']-F_D2)/1e9:+.1f} GHz beta={perf['beta_wg']:.3f} "
                  f"eta={perf['eta_atom']:.3f}")
        d_rows.append(_row("defect_scale", s_d, perf,
                           marker=("final" if abs(s_d - 1.0) < 1e-12 else ""),
                           selected=abs(s_d - 1.0) < 1e-12, hdf5=name))
    datasets.write_csv(paths.DATA_DIR / "tuning_defect_scale.csv", d_rows,
                       datasets.TUNING_COLUMNS)
    datasets.write_json(paths.DATA_DIR / "tuning_defect_scale.json",
                        {"rows": d_rows}, with_convention=True)

    # ── record the selected final candidate ─────────────────────────────────
    final_candidate = {
        "global_scale": s_sel,
        "source_label": paths.FINAL_SOURCE_LABEL,
        "n_cells": dict(par.n_cells),
        "context": dict(par.context),
        "layout": layout_ops.layout_to_dict(L_sel),
        "gate": gate,
        "perf_trim": perf_sel,
        "trim_hdf5": name_sel,
        "note": ("global in-plane scale trim; output-coupling rebalance NOT "
                 "applied" if gate["passed"] else
                 "global trim did not pass gate; see run_output_coupling"),
    }
    datasets.write_json(paths.FINAL_CANDIDATE_JSON, final_candidate,
                        with_convention=True)
    print(f"[trim] wrote {paths.FINAL_CANDIDATE_JSON}")
    if not gate["passed"]:
        print("[trim] NOTE: gate not fully passed — run_output_coupling.py "
              "will explore the β/η rebalance.")


if __name__ == "__main__":
    main()
