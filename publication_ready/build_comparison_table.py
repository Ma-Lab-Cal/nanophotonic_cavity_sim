"""Build the 5-design comparison table in one consistent convention.

design_1, invDes(design_1), seed, invDes(seed) recompute from saved HDF5 (zero
cloud cost); the trimmed "final" design is included if its verification sim
exists. Writes tables/comparison_table.csv and data/comparison_table.json.

    python -m publication_ready.build_comparison_table
"""

from __future__ import annotations

from publication_ready import paths
from publication_ready.lib import datasets, reuse

ORDER = ["design_1", "invDes(design_1)", "seed", "invDes(seed)", "final"]


def build() -> list:
    paths.ensure_dirs()
    rows = []
    for label in ORDER:
        if label in paths.HDF5_REGISTRY:
            src = paths.HDF5_REGISTRY[label]
        elif label == "final":
            src = paths.final_hdf5_path()
        else:
            continue
        if not src.exists():
            print(f"[compare] skip '{label}': {src} not found")
            continue
        print(f"[compare] recomputing perf for '{label}' from {src.name} ...")
        perf, _ = reuse.perf_from_hdf5(src)
        rows.append(datasets.perf_row(label, perf, source=str(src)))
        print(f"    f_res={perf['f_res_Hz']/1e12:.3f} THz  "
              f"det={(perf['f_res_Hz']-datasets.cfg.F_TARGET)/1e9:+.1f} GHz  "
              f"beta={perf['beta_wg']:.3f}  C={perf['C_atom']:.1f}  "
              f"eta={perf['eta_atom']:.3f}  Q={perf['Q']:.0f}")
    return rows


def main():
    rows = build()
    datasets.write_csv(paths.TABLES_DIR / "comparison_table.csv", rows,
                       datasets.COMPARISON_COLUMNS)
    datasets.write_json(paths.DATA_DIR / "comparison_table.json",
                        {"rows": rows}, with_convention=True)
    print(f"[compare] wrote {len(rows)} rows -> "
          f"{paths.TABLES_DIR/'comparison_table.csv'}")


if __name__ == "__main__":
    main()
