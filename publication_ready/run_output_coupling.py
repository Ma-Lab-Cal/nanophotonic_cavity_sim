"""Output-coupling sweep (Fig-5 Pareto axis; contingency rebalance if gate failed).

Sweeps the OUTPUT (left, −x) mirror hole-size scale around the trimmed
candidate to map the β–C–η overcoupling tradeoff. If the global trim did not
clear the β/η gate, additionally rebuilds the output mirror cell count and
re-trims, then re-selects the final candidate (argmax η subject to C≥20).

    python -m publication_ready.run_output_coupling
"""

from __future__ import annotations

import numpy as np

from inverse_design import config as cfg

from publication_ready import campaign, paths
from publication_ready.lib import datasets, layout_ops, reuse

F_D2 = cfg.F_TARGET


def _row(value, perf, hdf5=""):
    return {
        "knob": "output_holescale", "value": value,
        "f_res_THz": perf["f_res_Hz"] / 1e12,
        "detuning_GHz": (perf["f_res_Hz"] - F_D2) / 1e9,
        "Q": perf["Q"], "kappa_tot_2pi_GHz": perf["kappa_tot_2pi_GHz"],
        "beta_wg": perf["beta_wg"], "C_atom": perf["C_atom"],
        "eta_atom": perf["eta_atom"], "Vmode_lambda_n3": perf["Vmode_lambda_n3"],
        "g_2pi_GHz_at_250nm": perf["g_2pi_Hz_at_250nm"] / 1e9,
        "design_marker": "final" if abs(value - 1.0) < 1e-9 else "",
        "is_selected": abs(value - 1.0) < 1e-9, "hdf5": hdf5,
    }


def main():
    paths.ensure_dirs()
    fc = datasets.read_json(paths.FINAL_CANDIDATE_JSON)
    par = layout_ops.CavityParametrization(n_cells=fc["n_cells"],
                                           context=fc["context"])
    L_sel = layout_ops.layout_from_dict(fc["layout"], par)
    perf_sel = fc["perf_trim"]
    gate_passed = fc["gate"]["passed"]

    rows = []
    for m in campaign.OUTPUT_HOLESCALE:
        if abs(m - 1.0) < 1e-9:
            rows.append(_row(1.0, perf_sel, fc.get("trim_hdf5", "")))
            continue
        L = layout_ops.scale_output_mirror(L_sel, par, m)
        name = f"output_m{m:.2f}"
        perf, _, _ = reuse.verify_or_reuse(L, par, name, campaign.TRIM_RUN_TIME,
                                           campaign.TRIM_GRID_DL)
        rows.append(_row(m, perf, name))
        print(f"[output] m={m:.2f}: beta={perf['beta_wg']:.3f} "
              f"C={perf['C_atom']:.1f} eta={perf['eta_atom']:.3f} "
              f"det={(perf['f_res_Hz']-F_D2)/1e9:+.0f} GHz")

    datasets.write_csv(paths.DATA_DIR / "output_coupling_sweep.csv", rows,
                       datasets.TUNING_COLUMNS)
    datasets.write_json(paths.DATA_DIR / "output_coupling_sweep.json",
                        {"rows": rows}, with_convention=True)

    if gate_passed:
        print("[output] gate already passed at the trimmed candidate; "
              "hole-size sweep recorded for Fig. 5. No rebalance needed.")
        return

    # ── contingency: cell-count rebuild + re-trim + re-select ───────────────
    print("[output] gate not passed — escalating to output cell-count rebuild")
    candidates = []   # (eta, C, beta, detuning, layout, par, name, perf, s)
    layout0, _, par0 = layout_ops.load_layout_from_record(paths.FINAL_RECORD_DIR)
    s_sel = fc["global_scale"]
    for n_lm in campaign.OUTPUT_CELLCOUNTS:
        L_re, par_re = layout_ops.rebuild_with_output_cells(
            layout_ops.scale_layout(layout0, s_sel), par0, n_lm)
        # quick re-trim with a defect-only nudge to recover D2
        name = f"outcells_{n_lm}"
        perf, _, _ = reuse.verify_or_reuse(L_re, par_re, name,
                                           campaign.TRIM_RUN_TIME,
                                           campaign.TRIM_GRID_DL)
        sd = layout_ops.first_order_retune_factor(perf["f_res_Hz"])
        L_rt = layout_ops.scale_defect_region(L_re, par_re, 1.0 / sd)
        name2 = f"outcells_{n_lm}_retrim"
        perf2, _, _ = reuse.verify_or_reuse(L_rt, par_re, name2,
                                            campaign.TRIM_RUN_TIME,
                                            campaign.TRIM_GRID_DL)
        candidates.append((perf2["eta_atom"], perf2["C_atom"], perf2, L_rt,
                           par_re, name2, n_lm))
        print(f"[output] N_out={n_lm}: beta={perf2['beta_wg']:.3f} "
              f"C={perf2['C_atom']:.1f} eta={perf2['eta_atom']:.3f}")

    feasible = [c for c in candidates if c[1] >= campaign.GATE_C_MIN]
    pool = feasible or candidates
    best = max(pool, key=lambda c: c[0])
    _, _, perf2, L_rt, par_re, name2, n_lm = best
    fc.update({
        "n_cells": dict(par_re.n_cells),
        "context": dict(par_re.context),
        "layout": layout_ops.layout_to_dict(L_rt),
        "perf_trim": perf2,
        "trim_hdf5": name2,
        "gate": campaign.check_gate(perf2),
        "rebalanced_output_cells": n_lm,
        "note": f"output rebalance: N_left_mirror={n_lm}, argmax η s.t. C≥20",
    })
    datasets.write_json(paths.FINAL_CANDIDATE_JSON, fc, with_convention=True)
    print(f"[output] re-selected final: N_out={n_lm}, "
          f"eta={perf2['eta_atom']:.3f}, gate={fc['gate']['passed']}")


if __name__ == "__main__":
    main()
