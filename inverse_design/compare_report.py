"""Four-way comparison report:

  (1) design_1 baseline   (2) invDes(design_1)   (3) band-structure seed
  (4) invDes(seed)

Builds comparison figures from the two run records + the seed verification
data and assembles ``InverseDesign_Comparison_Report.pdf`` at the repo root.

Usage (repo root, tidyEnv):
    PYTHONPATH=. python -m inverse_design.compare_report
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from inverse_design import config as cfg
from inverse_design import record as rm
from inverse_design.make_report import Report, baseline_layout
from inverse_design.parametrization import CavityParametrization
from inverse_design.visualization import draw_layout_topview, layout_from_entry

RUN1 = cfg.RESULTS_ROOT / "full"
RUN2 = cfg.RESULTS_ROOT / "full_seed"
SEED_DIR = cfg.RESULTS_ROOT / "seed_design"
REP_DIR = RUN2 / "report"
PDF_PATH = cfg.REPO_ROOT / "InverseDesign_Comparison_Report.pdf"

LABELS = ["design_1 (baseline)", "invDes(design_1)", "seed (band-designed)",
          "invDes(seed)"]
COLORS = ["C3", "C1", "C2", "C0"]


# ──────────────────────────────────────────────────────────────────────────
# data loading
# ──────────────────────────────────────────────────────────────────────────

def seed_layout():
    seed = json.load(open(SEED_DIR / "seed.json"))
    params = {k: {"lattice": float(v["lattice"]),
                  "geometry_params": np.asarray(v["geometry_params"], dtype=float)}
              for k, v in seed["parameters"].items()}
    par = CavityParametrization(n_cells=seed["n_cells"])
    theta = par.init_theta_from_baseline(parameters=params,
                                         n_cells=seed["n_cells"])
    lay = CavityParametrization.detach(par.layout(theta))
    lay["vertices"] = [np.asarray(v) for v in lay["vertices"]]
    return lay


def load_all():
    rec1 = rm.load_record(RUN1)
    rec2 = rm.load_record(RUN2)
    base = json.load(open(RUN1 / "baseline.json"))["baseline_perf"]
    stage0_2 = json.load(open(RUN2 / "baseline.json"))
    seed_perf = stage0_2.get("seed_perf") or json.load(
        open(SEED_DIR / "seed_v2_perf.json"))
    perfs = [base,
             rec1[-1]["perf"]["diagnostic"],
             seed_perf,
             rec2[-1]["perf"]["diagnostic"]]
    return rec1, rec2, perfs


# ──────────────────────────────────────────────────────────────────────────
# figures
# ──────────────────────────────────────────────────────────────────────────

def fig_designs(rec1, rec2):
    lays = [baseline_layout(), layout_from_entry(rec1[-1])[0],
            seed_layout(), layout_from_entry(rec2[-1])[0]]
    ctx = dict(cfg.BASELINE_CONTEXT)
    fig, axes = plt.subplots(4, 1, figsize=(12, 8.6), tight_layout=True)
    for ax, lay, label in zip(axes, lays, LABELS):
        draw_layout_topview(lay, ctx, ax, title=label)
    p = REP_DIR / "cmp_designs.png"
    fig.savefig(p, dpi=200); plt.close(fig)
    return p


def fig_params(rec2):
    seed = seed_layout()
    opt, _ = layout_from_entry(rec2[-1])
    idx = np.arange(len(seed["a"]))
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.5), tight_layout=True,
                             sharex=True)
    for ax, key, label in zip(axes, ("a", "sx", "sy"),
                              ("lattice constant a$_i$ (µm)",
                               "hole x-scale s$_x$ (µm)",
                               "hole y-scale s$_y$ (µm)")):
        ax.plot(idx, np.asarray(seed[key]), "o--", ms=3, color="C2",
                label="seed", alpha=0.85)
        ax.plot(idx, np.asarray(opt[key]), "o-", ms=3, color="C0",
                label="invDes(seed)")
        ax.set_ylabel(label, fontsize=9)
        ax.legend(fontsize=8)
        for b in (5, 9, 49, 79):
            ax.axvline(b - 0.5, color="gray", lw=0.6, alpha=0.5)
    axes[0].set_title("Per-hole parameters, seed vs. optimized (sections: "
                      "5 taper | 4 output mirror | 40 defect | 30 back mirror "
                      "| 1 taper; output at left)", fontsize=10)
    axes[-1].set_xlabel("hole index (left → right)")
    p = REP_DIR / "cmp_params.png"
    fig.savefig(p, dpi=200); plt.close(fig)
    return p


_HIST_PANELS = [
    ("f_res_Hz", "f_res (THz)", 1e-12),
    ("eta_atom", "η (atom @250 nm)", 1.0),
    ("C_atom", "C (atom @250 nm)", 1.0),
    ("beta_wg", r"$\kappa_{wg}/\kappa_{tot}$", 1.0),
    ("Q", "Q (resonance finder)", 1.0),
    ("g_2pi_Hz_at_250nm", "g/2π @250 nm (GHz)", 1e-9),
]


def fig_histories(rec1, rec2, perfs):
    fig, axes = plt.subplots(3, 2, figsize=(11, 10.5), tight_layout=True)
    for ax, (key, label, scale) in zip(axes.ravel(), _HIST_PANELS):
        for rec, color, rlabel in ((rec1, "C1", "invDes(design_1)"),
                                   (rec2, "C0", "invDes(seed)")):
            pts = [(e["params"]["iteration"], e["perf"]["diagnostic"][key])
                   for e in rec if key in e.get("perf", {}).get("diagnostic", {})]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] * scale for p in pts],
                        "o-", color=color, label=rlabel)
        ax.axhline(perfs[0][key] * scale, ls="--", color="C3", lw=1.2,
                   label="design_1")
        ax.axhline(perfs[2][key] * scale, ls=":", color="C2", lw=1.4,
                   label="seed")
        if key == "f_res_Hz":
            ax.axhline(cfg.F_TARGET * scale, color="k", lw=0.8, alpha=0.6)
        if key in ("C_atom", "Q"):
            ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel(label)
        ax.legend(fontsize=7)
    fig.suptitle("Verified physical parameters vs. iteration — both runs "
                 "(dashed: design_1, dotted: seed)")
    p = REP_DIR / "cmp_histories.png"
    fig.savefig(p, dpi=200); plt.close(fig)
    return p


def fig_losses(rec1, rec2):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), tight_layout=True)
    for ax, rec, label, color in ((axes[0], rec1, "invDes(design_1)", "C1"),
                                  (axes[1], rec2, "invDes(seed)", "C0")):
        it = [e["params"]["iteration"] for e in rec]
        ax.plot(it, [e["loss"] for e in rec], "o-", color=color, ms=3)
        t = sum(e["time"]["duration_s"] for e in rec)
        ax.set_title(f"{label} — {len(rec)} iters, {t/3600:.2f} h", fontsize=10)
        ax.set_xlabel("iteration"); ax.set_ylabel("loss")
    p = REP_DIR / "cmp_losses.png"
    fig.savefig(p, dpi=200); plt.close(fig)
    return p


def fig_seed_bands():
    """Bands of the chosen seed unit cells (from seed_design/bands data)."""
    import pickle
    from inverse_design.bands import plot_bands
    with open(SEED_DIR / "bands.pkl", "rb") as f:
        res = pickle.load(f)
    keep = {k: res[k] for k in ("seed_defect", "seed_mirror") if k in res}
    titles = {"seed_defect": "Seed: defect cell",
              "seed_mirror": "Seed: mirror cell"}
    path = plot_bands(keep, REP_DIR, titles)
    (REP_DIR / "bands.png").rename(REP_DIR / "cmp_seed_bands.png")
    return REP_DIR / "cmp_seed_bands.png"


def fig_fields_final():
    import tidy3d as td
    sd = td.SimulationData.from_file(str(RUN2 / "verification" /
                                         "final_design.hdf5"))
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.4), tight_layout=True)
    for ax, mon, ydim, title in (
            (axes[0], "Field_Profile_Monitor_z", "y",
             "|E|, z = 0 (beam midplane) — invDes(seed) final design"),
            (axes[1], "Field_Profile_Monitor_y", "z",
             "|E|, y = 0 (vertical cut; atom 200–300 nm above top surface)")):
        data = sd[mon]
        arr = np.sqrt(sum(np.abs(getattr(data, c).isel(t=-1)) ** 2
                          for c in ("Ex", "Ey", "Ez"))).squeeze()
        im = ax.pcolormesh(arr.x, arr[ydim], arr.T, cmap="magma",
                           shading="auto")
        plt.colorbar(im, ax=ax, pad=0.01)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x (µm)"); ax.set_ylabel(f"{ydim} (µm)")
    p = REP_DIR / "cmp_fields.png"
    fig.savefig(p, dpi=200); plt.close(fig)
    return p


# ──────────────────────────────────────────────────────────────────────────
# PDF
# ──────────────────────────────────────────────────────────────────────────

def metric_table_rows(perfs):
    def fmt(v, spec):
        return spec.format(v) if v is not None else "—"
    rows = []
    specs = [
        ("f_res (THz)", "f_res_Hz", "{:.2f}", 1e-12),
        ("η (atom @250 nm)", "eta_atom", "{:.3f}", 1),
        ("C (atom @250 nm)", "C_atom", "{:.3g}", 1),
        ("β = κ_wg/κ_tot", "beta_wg", "{:.3f}", 1),
        ("Q (loaded)", "Q", "{:.3g}", 1),
        ("κ_tot/2π (GHz)", "kappa_tot_2pi_GHz", "{:.3g}", 1),
        ("g/2π @250 nm (GHz)", "g_2pi_Hz_at_250nm", "{:.2f}", 1e-9),
        ("V_mode ((λ/n)³)", "Vmode_lambda_n3", "{:.2f}", 1),
        ("Purcell", "purcell", "{:.3g}", 1),
    ]
    for label, key, spec, scale in specs:
        rows.append([label] + [fmt(p.get(key, None) and p[key] * scale, spec)
                               for p in perfs])
    return rows


def build_pdf(rec1, rec2, perfs):
    r = Report()
    pdf = r.pdf
    pdf.add_page()

    p1, p2, ps, p3 = perfs
    t1 = sum(e["time"]["duration_s"] for e in rec1)
    t2 = sum(e["time"]["duration_s"] for e in rec2)

    r.h1("Seeded inverse design of a SiN nanobeam cavity for the ⁸⁷Rb–waveguide "
         "interface: four-design comparison")
    r.p("Designs compared: (1) design_1 — the reproducible hand-tuned baseline; "
        "(2) invDes(design_1) — gradient optimization started from design_1 "
        f"({len(rec1)} iterations, {t1/3600:.2f} h); (3) seed — a classical "
        "band-structure design (mirror gap centered on the Rb D2 line, defect "
        "dielectric band edge just above it, Bragg-matched taper, one-sided "
        "4/30 mirror cells); (4) invDes(seed) — the same gradient optimization "
        f"started from the seed ({len(rec2)} iterations, {t2/3600:.2f} h). All "
        "numbers from the identical time-domain verification pipeline.", style="I")

    r.h2("1. Headline comparison")
    r.table_wide(["metric"] + LABELS, metric_table_rows(perfs))
    r.p(f"The seeded path wins decisively: η = {p3['eta_atom']:.3f} vs "
        f"{p2['eta_atom']:.3f} from the design_1 path "
        f"({p3['eta_atom']/p2['eta_atom']:.1f}×), and {p3['eta_atom']/p1['eta_atom']:.0f}× "
        "the original baseline. The two ingredients factorize cleanly: band-"
        "structure engineering supplies the photonics (seed Q = 4.7×10⁶, four "
        "thousand times design_1), and gradient inverse design supplies the "
        "interface (resonance pulled onto the D2 line and the output port "
        "opened to the overcoupled optimum).")

    pdf.add_page()
    r.h2("2. The four structures")
    r.img(REP_DIR / "cmp_designs.png",
          caption="Fig. 1 — Top views (z = 0). Output waveguide at left in all "
                  "cases.")
    r.img(REP_DIR / "cmp_params.png", w=r.W * 0.85,
          caption="Fig. 2 — Per-hole parameters: seed vs. invDes(seed).")

    pdf.add_page()
    r.h2("3. Seed unit-cell band structure")
    r.img(REP_DIR / "cmp_seed_bands.png", w=r.W * 0.8,
          caption="Fig. 3 — TE bands of the seed's mirror and defect cells "
                  "(Tidy3D Bloch sims). Mirror gap 356–412 THz centered on the "
                  "384.23 THz target (within 0.1 THz); defect dielectric edge "
                  "387.4 THz, so the resonance forms just below it, mid-gap of "
                  "the mirrors.")
    r.h2("4. Optimization trajectories")
    r.img(REP_DIR / "cmp_losses.png",
          caption="Fig. 4 — Loss vs. iteration for both runs.")
    pdf.add_page()
    r.img(REP_DIR / "cmp_histories.png", w=r.W * 0.92,
          caption="Fig. 5 — Verified physics vs. iteration for both runs; "
                  "dashed red: design_1, dotted green: seed.")

    pdf.add_page()
    r.h2("5. Final design fields")
    r.img(REP_DIR / "cmp_fields.png",
          caption="Fig. 6 — Cavity mode of invDes(seed), final verification.")

    r.h2("6. Discussion")
    add_discussion(r, perfs)

    r.h2("7. Reproducibility")
    r.p("Seed: invDesResults/seed_design/seed.json (band sweeps in "
        "seed_design/bands.json). Seeded run: 'python -m "
        "inverse_design.run_optimization --mode full --run-name full_seed "
        "--seed-file invDesResults/seed_design/seed.json'. Records with full "
        "reconstruction state: invDesResults/{full,full_seed}/record.json; "
        "structure-evolution GIFs and GDS in the same directories. This "
        "report: 'python -m inverse_design.compare_report'.", size=9)

    pdf.output(str(PDF_PATH))
    return PDF_PATH


def add_discussion(r, perfs):
    p1, p2, ps, p3 = perfs
    gamma = 6.0666e6
    C3 = p3["C_atom"]
    err3 = np.log(C3) / C3 if C3 > 1 else 1.0
    r.p("What band engineering bought. The seed's unit cells place the Rb D2 "
        "line mid-gap of the mirrors and just below the defect band edge — the "
        "textbook gentle-confinement prescription (Quan & Lončar 2011) — giving "
        f"Q = {ps['Q']:.2g} and V = {ps['Vmode_lambda_n3']:.2f} (λ/n)³ where "
        f"design_1 managed Q = {p1['Q']:.0f}. But raw Q is the wrong figure of "
        f"merit: at β = {ps['beta_wg']:.3f} the seed sends almost every photon "
        "into scattering or the back mirror rather than the waveguide, so its "
        f"interface efficiency (η = {ps['eta_atom']:.3f}) is no better than "
        "design_1's. The first seed iteration (7 output-mirror cells) made the "
        "point even more sharply: Q = 8.1×10⁶, C = 1.4×10⁵, η = 0.006.")
    r.p("What gradient optimization bought. Starting from design_1 (broken "
        "output photonics), the optimizer repaired the structure into a usable "
        f"interface (η = {p2['eta_atom']:.3f}) but stayed in a low-Q basin. "
        "Starting from the seed, the optimizer had only to retune the resonance "
        "and open the output port — exactly what the adjoint gradient sees "
        f"directly — reaching η = {p3['eta_atom']:.3f} at "
        f"f_res = {p3['f_res_Hz']/1e12:.2f} THz with C = {C3:.3g}.")
    r.p("Physical impact (literature context). With C in the "
        f"{'hundreds' if C3 > 100 else 'tens'}, the ln C/C atom–photon "
        f"entanglement error bound (Menon et al. 2020) is ≈{err3*100:.1f}% — "
        "C has ceased to be the limiting resource (demonstrated records are "
        "C ≈ 27–71; Suleymanzade & Dimitrova 2025). The interface figure that "
        "matters is η: two-node entanglement rates scale as η², so "
        f"η = {p1['eta_atom']:.3f} → {p3['eta_atom']:.3f} is a "
        f"{(p3['eta_atom']/p1['eta_atom'])**2:.0f}× rate gain over the original "
        "baseline. The remaining gap to η → β·η_fiber is set by the loss "
        "budget; the seeded design's β is the highest of the four and the "
        "design now operates with κ in the bad-cavity hierarchy κ ≫ g ≫ γ "
        "favored for fast, broadband photon extraction (Reiserer & Rempe 2015; "
        "Knall et al. 2022).", size=9.5)
    r.p("Caveats. Absolute g and V carry the legacy-convention factor-2 "
        "uncertainties noted in the first report; comparisons across the four "
        "designs are computed identically and are robust. The seed and its "
        "optimized variant inherit the 150 nm × 500 nm beam platform of "
        "design_1; thickness/width co-design (trap distance vs. photonics) "
        "remains open.", size=8.5)


def main():
    REP_DIR.mkdir(parents=True, exist_ok=True)
    rec1, rec2, perfs = load_all()
    fig_designs(rec1, rec2)
    fig_params(rec2)
    fig_histories(rec1, rec2, perfs)
    fig_losses(rec1, rec2)
    fig_seed_bands()
    fig_fields_final()
    print("pdf ->", build_pdf(rec1, rec2, perfs))


if __name__ == "__main__":
    main()
