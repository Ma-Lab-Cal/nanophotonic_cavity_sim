#!/usr/bin/env python
"""Generate the resonant-frequency convergence figure (Fig. 5) for the manuscript.

Reads the validation datasets produced by validate_design.py:
  publication_ready/data/validation/freq_resolution.json  (single-mesh ladder)
  publication_ready/data/validation/grid_offset.json      (grid-offset averaging + re-trim)
and renders a two-panel figure:
  (a) single-mesh f_res scatter vs mesh size (the grid-alignment problem),
  (b) grid-offset-averaged f_res, mesh-consistent within one linewidth, and the
      on-D2 calibrated re-trim.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
VAL = REPO / "publication_ready" / "data" / "validation"
F_D2_THZ = 299792458.0 * 1e6 / 0.780241 / 1e12      # matches inverse_design.config.F_TARGET
KAPPA = 31.0                                          # cavity linewidth (GHz), for the band

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9,
    "axes.labelsize": 9, "mathtext.fontset": "cm",
})


def _det(f_thz):
    return (np.asarray(f_thz) - F_D2_THZ) * 1000.0     # GHz


def main():
    freq = json.loads((VAL / "freq_resolution.json").read_text())
    go = json.loads((VAL / "grid_offset.json").read_text())["results"]

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.05, 2.9), constrained_layout=True)

    # ---- (a) single-mesh ladder scatter -------------------------------------
    nm = np.asarray(freq["ladder_nm"], float)
    det = _det(freq["ladder_fres_THz"])
    axa.axhspan(-KAPPA, KAPPA, color="tab:green", alpha=0.10,
                label=r"$\pm1$ linewidth")
    axa.axhline(0, color="0.6", lw=0.7)
    axa.plot(nm, det, "o-", color="tab:red", ms=5, label="single mesh")
    axa.set_xlabel(r"mesh size $\Delta\ell$ (nm)")
    axa.set_ylabel(r"$f_\mathrm{res}-f_\mathrm{D2}$ (GHz)")
    axa.set_title("(a) single-mesh scatter", loc="left")
    axa.invert_xaxis()
    axa.legend(fontsize=7, loc="lower left")

    # ---- (b) grid-offset averaging + re-trim --------------------------------
    axb.axhspan(-KAPPA, KAPPA, color="tab:green", alpha=0.10,
                label=r"$\pm1$ linewidth")
    axb.axhline(0, color="0.6", lw=0.7)
    colors = {"6": "tab:blue", "5": "tab:orange"}
    for key in ("6", "5"):
        r = go.get(key)
        if not r:
            continue
        offs = np.asarray(r["offsets_um"]) * 1e3            # nm
        d = _det(r["fres_THz"])
        c = colors[key]
        axb.scatter(offs, d, color=c, s=18, alpha=0.55,
                    label=f"{key} nm, per-offset")
        m = r["detuning_GHz"]
        axb.errorbar(offs.mean(), m, yerr=r["std_GHz"], fmt="D", color=c, ms=6,
                     capsize=3, label=f"{key} nm, offset-mean")
    # re-trimmed on-D2 point
    rt = go.get("retrim")
    if rt:
        axb.axhline(rt["mean_detuning_GHz"], color="k", lw=1.2)
        axb.errorbar(0.0, rt["mean_detuning_GHz"], yerr=rt.get("std_GHz", 0),
                     fmt="*", color="k", ms=13, capsize=3, zorder=6,
                     label=(f"re-trimmed: {rt['mean_detuning_GHz']:+.1f} GHz"))
    axb.set_xlabel("sub-cell structure offset (nm)")
    axb.set_ylabel(r"$f_\mathrm{res}-f_\mathrm{D2}$ (GHz)")
    axb.set_title("(b) grid-offset averaged + re-trim", loc="left")
    axb.legend(fontsize=6.2, loc="best", ncol=1, framealpha=0.9)

    out = REPO / "paper" / "figures" / "fig5_convergence.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
