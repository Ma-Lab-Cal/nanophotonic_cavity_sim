"""Final-candidate nominal verification: perf, loss budget, and field export.

Runs ONE high-resolution verification sim of the trimmed final candidate (or
reuses it), then writes the headline dataset:
  data/nominal_verify.json  — perf + loss budget + resonance spectrum + g/C/η
                              maps' nominal point + sim settings + convention
  data/fields_final.npz     — |E| and Re(Ey) on the three profile planes

    python -m publication_ready.nominal_verify            # cloud (1 sim)
    python -m publication_ready.nominal_verify --reuse-label invDes(seed)  # dev
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from inverse_design import config as cfg

from publication_ready import campaign, paths
from publication_ready.lib import datasets, fields_post, layout_ops, reuse


# ── field export ───────────────────────────────────────────────────────────

_PLANE_MONITORS = {
    "midplane_z": "Field_Profile_Monitor_z",     # xy at z=0
    "vertical_x": "Field_Profile_Monitor_x",     # yz at x=0
    "vertical_y": "Field_Profile_Monitor_y",     # xz at y=0
}


def _plane_arrays(mon):
    comps = {}
    for c in ("Ex", "Ey", "Ez"):
        a = getattr(mon, c, None)
        if a is None:
            continue
        if "t" in a.dims:
            a = a.isel(t=-1)
        if "f" in a.dims:
            a = a.isel(f=0)
        comps[c] = a.squeeze()
    if not comps:
        return None
    Eabs = np.sqrt(sum(np.abs(v) ** 2 for v in comps.values()))
    Ey = np.real(comps["Ey"]) if "Ey" in comps else np.zeros_like(np.real(Eabs))
    dims = list(Eabs.dims)
    axes = {d: np.asarray(Eabs[d], dtype=float) for d in dims}
    return Eabs, Ey, dims, axes


def export_fields(sim_obj) -> dict:
    out = {}
    md = sim_obj.sim_data.monitor_data
    for tag, mname in _PLANE_MONITORS.items():
        if mname not in md:
            continue
        # sim_data[name] applies symmetry expansion (full y,z extent);
        # .monitor_data[name] would only hold the y>=0 / z>=0 half.
        res = _plane_arrays(sim_obj.sim_data[mname])
        if res is None:
            continue
        Eabs, Ey, dims, axes = res
        out[f"{tag}_Eabs"] = np.asarray(Eabs, dtype=float)
        out[f"{tag}_Ey"] = np.asarray(Ey, dtype=float)
        out[f"{tag}_dims"] = np.array(dims)
        for d, vals in axes.items():
            out[f"{tag}_{d}"] = vals
    out["thickness_um"] = float(sim_obj.context["thickness"])
    out["width_um"] = float(sim_obj.context["width"])
    out["atom_height_um"] = float(sim_obj.context["thickness"] / 2 + 0.25)
    out["output_face"] = cfg.OUTPUT_FACE
    return out


def sim_settings(sim_obj) -> dict:
    sim = sim_obj.sim
    try:
        ng = int(np.prod(sim.grid.num_cells))
    except Exception:
        ng = None
    try:
        pml = sim.boundary_spec.x.minus.num_layers
    except Exception:
        pml = None
    return {
        "run_time_s": float(sim.run_time),
        "domain_size_um": [float(s) for s in sim.size],
        "num_grid_cells": ng,
        "pml_num_layers": pml,
        "grid_min_steps_per_wvl": 15,
    }


# ── analysis bundle ────────────────────────────────────────────────────────

def postprocess(perf, sim_obj) -> dict:
    budget = fields_post.loss_budget(sim_obj)
    spectrum = fields_post.resonance_spectrum(sim_obj)
    detuning_ghz = (perf["f_res_Hz"] - cfg.F_TARGET) / 1e9
    gate = campaign.check_gate(perf)
    return {
        "perf": perf,
        "detuning_GHz": detuning_ghz,
        "gate": gate,
        "loss_budget": budget,
        "resonance_spectrum": {
            "freq_Hz": spectrum["freq_Hz"], "Q": spectrum["Q"],
            "amplitude": spectrum["amplitude"], "error": spectrum["error"],
        },
        "sim_settings": sim_settings(sim_obj),
    }


def _load_final_candidate():
    with open(paths.FINAL_CANDIDATE_JSON) as f:
        fc = json.load(f)
    par = layout_ops.CavityParametrization(n_cells=fc["n_cells"],
                                           context=fc["context"])
    layout = layout_ops.layout_from_dict(fc["layout"], par)
    return fc, layout, par


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse-label", default=None,
                    help="dev: postprocess an existing design HDF5 instead of "
                         "running the trimmed final sim")
    args = ap.parse_args()
    paths.ensure_dirs()

    if args.reuse_label:
        perf, sim_obj = reuse.perf_for_label(args.reuse_label)
        tag = args.reuse_label
    else:
        fc, layout, par = _load_final_candidate()
        perf, sim_obj, _ = reuse.verify_or_reuse(
            layout, par, paths.FINAL_TRIMMED_NAME, campaign.FINAL_RUN_TIME,
            campaign.FINAL_GRID_DL)
        tag = "final_trimmed"

    bundle = postprocess(perf, sim_obj)
    bundle["design"] = tag
    datasets.write_json(paths.DATA_DIR / "nominal_verify.json", bundle,
                        with_convention=True)

    fields = export_fields(sim_obj)
    datasets.write_npz(paths.DATA_DIR / "fields_final.npz", **fields)

    # final-design geometry package (only for the real trimmed candidate)
    if not args.reuse_label:
        from publication_ready.lib import geometry_export
        fc, layout, par = _load_final_candidate()
        pkg = geometry_export.export_package(layout, par.context, par.n_cells,
                                             paths.DATA_DIR, perf=perf)
        print(f"[nominal] geometry package: {pkg['csv'].name}, "
              f"{pkg['json'].name}, {pkg['gds'].name if pkg['gds'] else 'no-gds'}")

    b = bundle["loss_budget"]
    print(f"[nominal] {tag}: f_res={perf['f_res_Hz']/1e12:.4f} THz "
          f"({bundle['detuning_GHz']:+.1f} GHz)  Q={perf['Q']:.0f}")
    print(f"[nominal] beta={perf['beta_wg']:.3f}  C={perf['C_atom']:.1f}  "
          f"eta={perf['eta_atom']:.3f}  V={perf['Vmode_lambda_n3']:.3f}")
    print(f"[nominal] loss budget: kappa_wg={b['kappa_wg_2pi_GHz']:.2f} GHz  "
          f"residual={b['kappa_residual_2pi_GHz']:.2f} GHz  "
          f"gate_passed={bundle['gate']['passed']}")
    print(f"[nominal] wrote nominal_verify.json + fields_final.npz")


if __name__ == "__main__":
    main()
