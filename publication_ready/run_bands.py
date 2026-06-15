"""Bloch band structure for Fig. 2 (final cells computed; seed cells reused).

The seed unit-cell bands already exist (invDesResults/seed_design/bands.json);
we subset a representative mirror + defect cell from them and recompute the
final design's mirror + defect cells (18 cheap Bloch sims). Writes:
  data/bands_final.json   (final_mirror, final_defect)
  data/bands_seed.json    (seed_mirror, seed_defect)

    python -m publication_ready.run_bands
    python -m publication_ready.run_bands --reuse-record   # final from invDes(seed)
"""

from __future__ import annotations

import argparse

import numpy as np

from inverse_design import bands as bands_mod
from inverse_design import config as cfg

from publication_ready import paths
from publication_ready.lib import datasets, layout_ops


def _final_layout_par(reuse_record):
    if paths.FINAL_CANDIDATE_JSON.exists() and not reuse_record:
        fc = datasets.read_json(paths.FINAL_CANDIDATE_JSON)
        par = layout_ops.CavityParametrization(n_cells=fc["n_cells"],
                                               context=fc["context"])
        layout = layout_ops.layout_from_dict(fc["layout"], par)
        return layout, dict(fc["context"]), par
    return layout_ops.load_layout_from_record(paths.FINAL_RECORD_DIR)


def _seed_subset():
    """Pick one mirror + one defect cell from the saved seed band sweep."""
    if not paths.SEED_BANDS_JSON.exists():
        print(f"[bands] seed bands {paths.SEED_BANDS_JSON} not found; skipping")
        return {}
    seed = datasets.read_json(paths.SEED_BANDS_JSON)
    out = {}
    mir = [k for k in seed if k.startswith("mir")]
    deff = [k for k in seed if k.startswith("def")]
    pick_mir = "mir_a336" if "mir_a336" in seed else (mir[len(mir)//2] if mir else None)
    pick_def = "def_a297" if "def_a297" in seed else (deff[len(deff)//2] if deff else None)
    if pick_mir:
        out["seed_mirror"] = seed[pick_mir]
    if pick_def:
        out["seed_defect"] = seed[pick_def]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse-record", action="store_true")
    args = ap.parse_args()
    paths.ensure_dirs()
    cfg.ensure_api_key()

    layout, context, par = _final_layout_par(args.reuse_record)
    a = np.asarray(layout["a"])
    nc = par.n_cells
    i_def = nc["N_left_taper"] + nc["N_left_mirror"] + nc["N_defect"] // 2
    i_mir = (nc["N_left_taper"] + nc["N_left_mirror"] + nc["N_defect"]
             + nc["N_right_mirror"] // 2)
    cells = {
        "final_defect": (float(a[i_def]), np.asarray(layout["vertices"][i_def])),
        "final_mirror": (float(a[i_mir]), np.asarray(layout["vertices"][i_mir])),
    }
    print(f"[bands] final cells: defect a={a[i_def]*1e3:.1f} nm (hole {i_def}), "
          f"mirror a={a[i_mir]*1e3:.1f} nm (hole {i_mir})")

    out_dir = paths.VERIFY_DIR / "bands_final"
    out_dir.mkdir(parents=True, exist_ok=True)
    bands_mod.compute_bands(cells, context, out_dir, folder_name="pub_bands_final")
    final_bands = datasets.read_json(out_dir / "bands.json")
    # keep only the freshly computed final cells
    final_bands = {k: v for k, v in final_bands.items() if k.startswith("final_")}
    datasets.write_json(paths.DATA_DIR / "bands_final.json", final_bands)

    seed_bands = _seed_subset()
    if seed_bands:
        datasets.write_json(paths.DATA_DIR / "bands_seed.json", seed_bands)
    print(f"[bands] wrote bands_final.json ({list(final_bands)}) "
          f"+ bands_seed.json ({list(seed_bands)})")


if __name__ == "__main__":
    main()
