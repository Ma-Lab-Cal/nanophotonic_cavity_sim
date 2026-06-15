"""Numerical convergence study on the final candidate (mesh / PML / run-time).

Each axis shares the final high-res sim as its baseline point (reused, no extra
sim). Reports the drift of f_res, Q, β, η, V. Writes:
  data/convergence_mesh.csv, convergence_pml.csv, convergence_runtime.csv

    python -m publication_ready.run_convergence
"""

from __future__ import annotations

from inverse_design import config as cfg

from publication_ready import campaign, paths
from publication_ready.lib import datasets, layout_ops, reuse

F_D2 = cfg.F_TARGET


def _row(axis, value, perf):
    return {
        "axis": axis, "value": value,
        "f_res_THz": perf["f_res_Hz"] / 1e12,
        "detuning_GHz": (perf["f_res_Hz"] - F_D2) / 1e9,
        "Q": perf["Q"], "kappa_tot_2pi_GHz": perf["kappa_tot_2pi_GHz"],
        "beta_wg": perf["beta_wg"], "C_atom": perf["C_atom"],
        "eta_atom": perf["eta_atom"], "Vmode_lambda_n3": perf["Vmode_lambda_n3"],
    }


def main():
    paths.ensure_dirs()
    fc = datasets.read_json(paths.FINAL_CANDIDATE_JSON)
    par = layout_ops.CavityParametrization(n_cells=fc["n_cells"],
                                           context=fc["context"])
    layout = layout_ops.layout_from_dict(fc["layout"], par)
    base_perf, _ = reuse.perf_from_hdf5(paths.final_hdf5_path())
    base_dl = campaign.FINAL_GRID_DL[0]

    # build batch specs for all non-baseline convergence points
    specs = []
    for dl in campaign.CONV_MESH_DL:
        if abs(dl - base_dl) < 1e-9:
            continue
        specs.append({"name": f"conv_mesh_{dl:.3f}".replace(".", "p"),
                      "layout": layout, "par": par, "grid_dl": (dl, dl, dl),
                      "run_time": campaign.FINAL_RUN_TIME})
    for n in campaign.CONV_PML_LAYERS:
        if n == 12:
            continue
        specs.append({"name": f"conv_pml_{n}", "layout": layout, "par": par,
                      "grid_dl": campaign.FINAL_GRID_DL, "pml_layers": n,
                      "run_time": campaign.FINAL_RUN_TIME})
    for rt in campaign.CONV_RUNTIME:
        if abs(rt - campaign.FINAL_RUN_TIME) < 1e-15:
            continue
        specs.append({"name": f"conv_rt_{rt*1e12:.0f}ps", "layout": layout,
                      "par": par, "grid_dl": campaign.FINAL_GRID_DL,
                      "run_time": rt})
    batch = reuse.verify_batch(specs, campaign.FINAL_RUN_TIME,
                               campaign.FINAL_GRID_DL, folder="conv_batch")

    def _perf(name):
        r = batch.get(name)
        return r[0] if r else None          # None if the sim diverged/failed

    mesh_rows = []
    for dl in campaign.CONV_MESH_DL:
        perf = base_perf if abs(dl - base_dl) < 1e-9 else \
            _perf(f"conv_mesh_{dl:.3f}".replace(".", "p"))
        if perf:
            mesh_rows.append(_row("mesh_dl_um", dl, perf))
    datasets.write_csv(paths.DATA_DIR / "convergence_mesh.csv", mesh_rows,
                       datasets.CONVERGENCE_COLUMNS)

    pml_rows = []
    for n in campaign.CONV_PML_LAYERS:
        perf = base_perf if n == 12 else _perf(f"conv_pml_{n}")
        if perf:
            pml_rows.append(_row("pml_layers", n, perf))
    datasets.write_csv(paths.DATA_DIR / "convergence_pml.csv", pml_rows,
                       datasets.CONVERGENCE_COLUMNS)

    rt_rows = []
    for rt in campaign.CONV_RUNTIME:
        perf = base_perf if abs(rt - campaign.FINAL_RUN_TIME) < 1e-15 else \
            _perf(f"conv_rt_{rt*1e12:.0f}ps")
        if perf:
            rt_rows.append(_row("run_time_s", rt, perf))
    datasets.write_csv(paths.DATA_DIR / "convergence_runtime.csv", rt_rows,
                       datasets.CONVERGENCE_COLUMNS)
    print(f"[conv] wrote convergence csvs: mesh={len(mesh_rows)} "
          f"pml={len(pml_rows)} runtime={len(rt_rows)} rows")


if __name__ == "__main__":
    main()
