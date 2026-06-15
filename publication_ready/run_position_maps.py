"""Atom-position robustness maps (postprocess of the final sim; zero cloud).

η and C versus atom height (200-350 nm) and lateral (y) / along-beam (x) offset,
with κ and β held from the cavity. Writes:
  data/position_map_yz.npz  — (z height × y lateral)
  data/position_map_xz.npz  — (z height × x along-beam)

    python -m publication_ready.run_position_maps
    python -m publication_ready.run_position_maps --reuse-label invDes(seed)
"""

from __future__ import annotations

import argparse

from publication_ready import campaign, paths
from publication_ready.lib import datasets, fields_post, reuse


def _final_perf_sim(reuse_label):
    if reuse_label:
        return reuse.perf_for_label(reuse_label)
    return reuse.perf_from_hdf5(paths.final_hdf5_path())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse-label", default=None)
    args = ap.parse_args()
    paths.ensure_dirs()

    perf, sim = _final_perf_sim(args.reuse_label)
    beta = perf["beta_wg"]

    yz = fields_post.g_C_eta_grid(sim, beta, campaign.POS_Z_HEIGHTS_NM,
                                  campaign.POS_Y_OFFSETS_NM)
    datasets.write_npz(paths.DATA_DIR / "position_map_yz.npz", **yz)

    # along-beam offsets: reuse the lateral grid magnitudes as x offsets
    xz = fields_post.g_C_eta_grid_axial(sim, beta, campaign.POS_Z_HEIGHTS_NM,
                                        campaign.POS_Y_OFFSETS_NM)
    datasets.write_npz(paths.DATA_DIR / "position_map_xz.npz", **xz)

    i_nom_z = list(campaign.POS_Z_HEIGHTS_NM).index(250)
    i_nom_y = list(campaign.POS_Y_OFFSETS_NM).index(0)
    print(f"[posmap] beta held = {beta:.3f}")
    print(f"[posmap] nominal (250nm,0): eta={yz['eta'][i_nom_z,i_nom_y]:.3f}  "
          f"C={yz['C'][i_nom_z,i_nom_y]:.1f}  "
          f"g/2pi={yz['g_2pi_Hz'][i_nom_z,i_nom_y]/1e9:.3f} GHz")
    print(f"[posmap] eta range over heights at y=0: "
          f"{yz['eta'][:,i_nom_y].min():.3f}..{yz['eta'][:,i_nom_y].max():.3f}")
    print("[posmap] wrote position_map_yz.npz + position_map_xz.npz")


if __name__ == "__main__":
    main()
