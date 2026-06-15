"""Fabrication-tolerance robustness: deterministic corner grid.

Perturbs the final candidate by etch-bias (hole radius), lattice-constant, and
beam-width biases at low/high corners + single-axis extremes (~15 sims). Each
sample stores the unretuned metrics and the analytic global-scale retune that
restores D2 (β/η are ≈ preserved by that scale, the published robustness claim).
Writes:
  data/tolerance_sweep.csv  + data/tolerance_samples/<id>.json

    python -m publication_ready.run_tolerance
"""

from __future__ import annotations

from inverse_design import config as cfg

from publication_ready import campaign, paths
from publication_ready.lib import datasets, layout_ops, reuse

F_D2 = cfg.F_TARGET


def _sample_id(pt):
    return (f"e{pt['etch_nm']:+.0f}_l{pt['lattice_nm']:+.0f}"
            f"_w{pt['width_nm']:+.0f}")


def main():
    paths.ensure_dirs()
    fc = datasets.read_json(paths.FINAL_CANDIDATE_JSON)
    par = layout_ops.CavityParametrization(n_cells=fc["n_cells"],
                                           context=fc["context"])
    layout = layout_ops.layout_from_dict(fc["layout"], par)

    grid = campaign.tolerance_corner_grid()

    # build batch specs for the non-nominal corners (parallel cloud wall-clock)
    specs, meta = [], {}
    for pt in grid:
        sid = _sample_id(pt)
        if pt["etch_nm"] == 0 and pt["lattice_nm"] == 0 and pt["width_nm"] == 0:
            continue
        L = layout_ops.perturb_layout(layout, par, pt["etch_nm"], pt["lattice_nm"])
        if pt["width_nm"]:
            ctx = layout_ops.context_with_width(fc["context"], pt["width_nm"])
            par_v = layout_ops.CavityParametrization(n_cells=fc["n_cells"],
                                                     context=ctx)
        else:
            par_v = par
        specs.append({"name": f"tol_{sid}", "layout": L, "par": par_v})
        meta[sid] = pt
    batch = reuse.verify_batch(specs, campaign.TRIM_RUN_TIME,
                               campaign.TRIM_GRID_DL, folder="tol_batch")
    nominal_perf, _ = reuse.perf_from_hdf5(paths.final_hdf5_path())

    rows = []
    for pt in grid:
        sid = _sample_id(pt)
        is_nominal = (pt["etch_nm"] == 0 and pt["lattice_nm"] == 0
                      and pt["width_nm"] == 0)
        perf = nominal_perf if is_nominal else batch[f"tol_{sid}"][0]

        f_res = perf["f_res_Hz"]
        det_un = (f_res - F_D2) / 1e9
        retune_scale = layout_ops.first_order_retune_factor(f_res)
        det_re = (f_res / retune_scale - F_D2) / 1e9     # ≈ 0 by construction
        row = {
            "sample_id": sid, "mode": "corner",
            "etch_nm": pt["etch_nm"], "lattice_nm": pt["lattice_nm"],
            "width_nm": pt["width_nm"], "f_res_THz": f_res / 1e12,
            "detuning_unretuned_GHz": det_un, "detuning_retuned_GHz": det_re,
            "retune_scale": retune_scale, "Q": perf["Q"],
            "beta_wg": perf["beta_wg"], "C_atom": perf["C_atom"],
            "eta_atom": perf["eta_atom"], "hdf5": f"tol_{sid}",
        }
        rows.append(row)
        datasets.write_json(paths.TOL_SAMPLES_DIR / f"{sid}.json",
                            {"perturbation": pt, "perf": perf, "row": row})
        print(f"[tol] {sid}: det_un={det_un:+.0f} GHz beta={perf['beta_wg']:.3f} "
              f"eta={perf['eta_atom']:.3f} C={perf['C_atom']:.1f}")

    datasets.write_csv(paths.DATA_DIR / "tolerance_sweep.csv", rows,
                       datasets.TOLERANCE_COLUMNS)
    etas = [r["eta_atom"] for r in rows]
    betas = [r["beta_wg"] for r in rows]
    print(f"[tol] {len(rows)} samples; eta {min(etas):.3f}..{max(etas):.3f}, "
          f"beta {min(betas):.3f}..{max(betas):.3f}")


if __name__ == "__main__":
    main()
