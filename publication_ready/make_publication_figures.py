"""Build all publication figures from saved datasets (no live sims).

Every figure depends only on files under publication_ready/data and
publication_ready/tables, so this is free and deterministic. Figures whose
backing data is not present are skipped with a notice.

    python -m publication_ready.make_publication_figures            # all
    python -m publication_ready.make_publication_figures --only fig5
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from publication_ready import paths
from publication_ready.lib import datasets, layout_ops, plots


def _final_layout():
    """Final trimmed layout if available, else invDes(seed) for development."""
    if paths.FINAL_CANDIDATE_JSON.exists():
        fc = datasets.read_json(paths.FINAL_CANDIDATE_JSON)
        par = layout_ops.CavityParametrization(n_cells=fc["n_cells"],
                                               context=fc["context"])
        layout = layout_ops.layout_from_dict(fc["layout"], par)
        return layout, fc["context"]
    layout, ctx, _ = layout_ops.load_layout_from_record(paths.FINAL_RECORD_DIR)
    print("[figs] final_candidate.json missing; using invDes(seed) for Fig.1")
    return layout, ctx


FIGS = {}


def register(name):
    def deco(fn):
        FIGS[name] = fn
        return fn
    return deco


@register("fig1")
def fig1(_):
    final_layout, ctx = _final_layout()
    seed_layout, _, _ = layout_ops.load_seed_layout()
    return plots.fig_device_and_traces(final_layout, ctx, seed_layout,
                                       paths.FIG_MAIN / "Fig1_device.pdf")


@register("fig2")
def fig2(_):
    bands = {}
    for path in (paths.DATA_DIR / "bands_final.json",
                 paths.DATA_DIR / "bands_seed.json"):
        if path.exists():
            bands.update(datasets.read_json(path))
    if not bands:
        print("[figs] fig2 skipped: no band data (run run_bands.py)")
        return None
    f_res = None
    if (paths.DATA_DIR / "nominal_verify.json").exists():
        nv = datasets.read_json(paths.DATA_DIR / "nominal_verify.json")
        f_res = nv["perf"]["f_res_Hz"] / 1e12
    return plots.fig_bands(bands, paths.FIG_MAIN / "Fig2_bands.pdf",
                           f_res_thz=f_res)


@register("fig3")
def fig3(_):
    p = paths.DATA_DIR / "nominal_verify.json"
    if not p.exists():
        print("[figs] fig3 skipped: run nominal_verify.py")
        return None
    return plots.fig_resonance_loss(datasets.read_json(p),
                                    paths.FIG_MAIN / "Fig3_resonance_loss.pdf")


@register("fig4")
def fig4(_):
    p = paths.DATA_DIR / "fields_final.npz"
    if not p.exists():
        print("[figs] fig4 skipped: run nominal_verify.py")
        return None
    fields = dict(np.load(p, allow_pickle=True))
    return plots.fig_fields(fields, paths.FIG_MAIN / "Fig4_fields.pdf")


@register("fig5")
def fig5(_):
    def _rows(name):
        p = paths.DATA_DIR / name
        return datasets.read_csv(p) if p.exists() else []
    g = _rows("tuning_global_scale.csv")
    d = _rows("tuning_defect_scale.csv")
    o = _rows("output_coupling_sweep.csv")
    if not g:
        print("[figs] fig5 skipped: run run_tuning_sweeps.py")
        return None
    return plots.fig_tuning(g, d, o, paths.FIG_MAIN / "Fig5_tuning.pdf")


@register("fig6")
def fig6(_):
    pyz = paths.DATA_DIR / "position_map_yz.npz"
    pxz = paths.DATA_DIR / "position_map_xz.npz"
    if not pyz.exists():
        print("[figs] fig6 skipped: run run_position_maps.py")
        return None
    yz = dict(np.load(pyz))
    xz = dict(np.load(pxz)) if pxz.exists() else yz
    return plots.fig_position_heatmaps(yz, xz,
                                       paths.FIG_MAIN / "Fig6_position.pdf")


@register("fig7")
def fig7(_):
    p = paths.DATA_DIR / "tolerance_sweep.csv"
    if not p.exists():
        print("[figs] fig7 skipped: run run_tolerance.py")
        return None
    return plots.fig_fab_robustness(datasets.read_csv(p),
                                    paths.FIG_MAIN / "Fig7_fabrication.pdf")


@register("fig8")
def fig8(_):
    p = paths.TABLES_DIR / "comparison_table.csv"
    if not p.exists():
        print("[figs] fig8 skipped: run build_comparison_table.py")
        return None
    return plots.fig_comparison_bars(datasets.read_csv(p),
                                     paths.FIG_MAIN / "Fig8_comparison.pdf")


@register("supp_convergence")
def supp_convergence(_):
    def _rows(name):
        p = paths.DATA_DIR / name
        return datasets.read_csv(p) if p.exists() else []
    m = _rows("convergence_mesh.csv")
    pm = _rows("convergence_pml.csv")
    rt = _rows("convergence_runtime.csv")
    if not (m or pm or rt):
        print("[figs] supp_convergence skipped: run run_convergence.py")
        return None
    return plots.fig_convergence(m, pm, rt,
                                 paths.FIG_SUPP / "SuppA_convergence.pdf")


@register("supp_modeiso")
def supp_modeiso(_):
    csv_p = paths.DATA_DIR / "mode_scan.csv"
    npz_p = paths.DATA_DIR / "competing_modes.npz"
    if not csv_p.exists() or not npz_p.exists():
        print("[figs] supp_modeiso skipped: run run_mode_scan.py")
        return None
    rows = datasets.read_csv(csv_p)
    d = dict(np.load(npz_p))
    return plots.fig_mode_isolation(
        d.get("spectrum_freq_Hz", np.array([])), d.get("spectrum_amp", np.array([])),
        rows, float(d["f_selected_Hz"][0]) / 1e12, float(d["kappa_2pi_Hz"][0]),
        paths.FIG_SUPP / "SuppB_mode_isolation.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="fig1..fig8")
    args = ap.parse_args()
    paths.ensure_dirs()
    names = [args.only] if args.only else list(FIGS)
    for name in names:
        try:
            out = FIGS[name](None)
            if out:
                print(f"[figs] {name} -> {out}")
        except Exception as exc:
            import traceback
            print(f"[figs] {name} FAILED: {exc}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
