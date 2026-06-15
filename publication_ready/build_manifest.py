"""Build the data/figure manifest + inventory with checksums.

Writes publication_ready_manifest.md (figure → backing dataset → command) and
inventory.md (every data/figure file with size + sha256), so each figure is
traceable to the exact file and command that regenerate it.

    python -m publication_ready.build_manifest
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from publication_ready import paths

PUB = paths.PUB_ROOT

# figure -> (what it shows, backing datasets, regenerate cmd, needs --recompute)
FIGURE_MANIFEST = [
    ("Fig1_device", "Top-view geometry + per-hole a/sₓ/s_y traces (seed vs final)",
     ["data/final_candidate.json", "seeds/seed_780nm.json"],
     "make_publication_figures --only fig1", "no"),
    ("Fig2_bands", "Bloch bands of mirror & defect cells; gap, light line, D2, f_res",
     ["data/bands_final.json", "data/bands_seed.json"],
     "make_publication_figures --only fig2", "bands only"),
    ("Fig3_resonance_loss", "Resonance fit (Q, κ) + κ_wg/residual loss budget",
     ["data/nominal_verify.json"],
     "make_publication_figures --only fig3", "no"),
    ("Fig4_fields", "|E| & Re(E_y) field cuts + field at the atom height",
     ["data/fields_final.npz"],
     "make_publication_figures --only fig4", "no"),
    ("Fig5_tuning", "Global/defect/output sweeps; overcoupling Pareto",
     ["data/tuning_global_scale.csv", "data/tuning_defect_scale.csv",
      "data/output_coupling_sweep.csv"],
     "make_publication_figures --only fig5", "yes"),
    ("Fig6_position", "η & C vs atom height and lateral / along-beam offset",
     ["data/position_map_yz.npz", "data/position_map_xz.npz"],
     "make_publication_figures --only fig6", "no"),
    ("Fig7_fabrication", "η/β/detuning/Q under fabrication corner perturbations",
     ["data/tolerance_sweep.csv"],
     "make_publication_figures --only fig7", "yes"),
    ("Fig8_comparison", "Five-design comparison bars (one convention)",
     ["tables/comparison_table.csv"],
     "make_publication_figures --only fig8", "no"),
    ("SuppA_convergence", "Mesh / PML / run-time convergence",
     ["data/convergence_mesh.csv", "data/convergence_pml.csv",
      "data/convergence_runtime.csv"],
     "make_publication_figures --only supp_convergence", "yes"),
    ("SuppB_mode_isolation", "Spectrum + nearest-competitor separation",
     ["data/mode_scan.csv", "data/competing_modes.npz"],
     "make_publication_figures --only supp_modeiso", "no"),
]

SOURCE_HDF5 = {
    "design_1": "invDesResults/full_seed/verification/baseline_design1.hdf5",
    "invDes(design_1)": "invDesResults/full/verification/final_design.hdf5",
    "seed": "invDesResults/full_seed/verification/seed_design.hdf5",
    "invDes(seed)": "invDesResults/full_seed/verification/final_design.hdf5",
    "final (trimmed)": "publication_ready/verification/final_trimmed.hdf5",
}


def _sha256(path, limit=64):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:limit]


def _exists(rel):
    return (paths.REPO_ROOT / rel).exists()


def build_manifest_md() -> str:
    L = ["# Publication package manifest", "",
         "Every figure regenerates from a saved dataset with one command. "
         "`./publication_ready/reproduce_all.sh` (no flag) rebuilds all figures "
         "and tables from saved data with **zero cloud cost**; "
         "`--recompute` re-runs the cloud sweeps/sims.", "",
         "## Figures", "",
         "| Figure | Shows | Backing dataset(s) | Regenerate (python -m publication_ready.<cmd>) | Needs --recompute |",
         "|---|---|---|---|---|"]
    for name, shows, data, cmd, recomp in FIGURE_MANIFEST:
        present = "✓" if (paths.FIG_MAIN / f"{name}.pdf").exists() or \
                          (paths.FIG_SUPP / f"{name}.pdf").exists() else "—"
        data_s = "<br>".join(f"`{d}`{'' if _exists('publication_ready/'+d) or _exists(d) else ' (missing)'}"
                             for d in data)
        L.append(f"| {name} {present} | {shows} | {data_s} | `{cmd}` | {recomp} |")
    L += ["", "## Source verification simulations (provenance)", "",
          "| Design | HDF5 | present |", "|---|---|---|"]
    for label, rel in SOURCE_HDF5.items():
        L.append(f"| {label} | `{rel}` | {'✓' if _exists(rel) else '—'} |")
    L += ["", "## Datasets", "",
          "| File | Description |", "|---|---|",
          "| `tables/comparison_table.csv` | 5-design metrics, one convention |",
          "| `data/nominal_verify.json` | final perf + loss budget + spectrum + settings |",
          "| `data/fields_final.npz` | field cuts on the three profile planes |",
          "| `data/tuning_global_scale.csv` | global-scale resonance trim axis |",
          "| `data/tuning_defect_scale.csv` | defect-only fine-trim axis |",
          "| `data/output_coupling_sweep.csv` | output mirror β–C–η Pareto axis |",
          "| `data/position_map_{yz,xz}.npz` | η/C vs atom height & offset |",
          "| `data/tolerance_sweep.csv` | fabrication corner-grid robustness |",
          "| `data/convergence_{mesh,pml,runtime}.csv` | numerical convergence |",
          "| `data/mode_scan.csv` + `competing_modes.npz` | mode isolation |",
          "| `data/bands_{final,seed}.json` | Bloch unit-cell bands |",
          "| `data/final_design.{json,csv,gds}` | fabrication geometry package |",
          ""]
    return "\n".join(L)


def build_inventory_md() -> str:
    L = ["# Inventory (size + sha256 of every data/figure artifact)", "",
         "| File | bytes | sha256[:16] |", "|---|---|---|"]
    roots = [paths.DATA_DIR, paths.TABLES_DIR, paths.FIG_MAIN, paths.FIG_SUPP]
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix in (".csv", ".json", ".npz", ".pdf",
                                            ".gds", ".png"):
                rel = p.relative_to(PUB)
                L.append(f"| `{rel}` | {p.stat().st_size} | "
                         f"`{_sha256(p, 16)}` |")
    return "\n".join(L)


def main():
    paths.ensure_dirs()
    (PUB / "publication_ready_manifest.md").write_text(build_manifest_md())
    (PUB / "inventory.md").write_text(build_inventory_md())
    print(f"[manifest] wrote {PUB/'publication_ready_manifest.md'}")
    print(f"[manifest] wrote {PUB/'inventory.md'}")


if __name__ == "__main__":
    main()
