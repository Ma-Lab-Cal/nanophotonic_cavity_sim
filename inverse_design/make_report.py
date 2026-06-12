"""Generate the PDF design report for the inverse-design run.

Builds all report figures from the saved record / verification data in
``invDesResults/full/`` (plus the band diagrams from ``inverse_design.bands``)
and assembles ``InverseDesign_Report.pdf`` at the repo root.

Usage (repo root, tidyEnv):
    PYTHONPATH=. python -m inverse_design.make_report
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
from inverse_design.visualization import draw_layout_topview, layout_from_entry

RUN_DIR = cfg.RESULTS_ROOT / "full"
REP_DIR = RUN_DIR / "report"
PDF_PATH = cfg.REPO_ROOT / "InverseDesign_Report.pdf"

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


# ──────────────────────────────────────────────────────────────────────────
# Figure builders
# ──────────────────────────────────────────────────────────────────────────

def baseline_layout():
    from main_code.cavity import Cavity
    from inverse_design.parametrization import CavityParametrization

    par = CavityParametrization()
    theta0 = par.init_theta_from_baseline()
    lay = CavityParametrization.detach(par.layout(theta0))
    lay["vertices"] = [np.asarray(v) for v in lay["vertices"]]
    return lay


def fig_design(rec):
    layout, context = layout_from_entry(rec[-1])
    base = baseline_layout()

    fig, axes = plt.subplots(2, 1, figsize=(12, 4.6), tight_layout=True)
    draw_layout_topview(base, context, axes[0],
                        title="Baseline (design_1) — top view, z = 0")
    draw_layout_topview(layout, context, axes[1],
                        title="Inverse-designed — top view, z = 0")
    p1 = REP_DIR / "fig_design_full.png"
    fig.savefig(p1, dpi=200); plt.close(fig)

    # zoom near the defect
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.6), tight_layout=True)
    for ax, lay, name in ((axes[0], base, "baseline"),
                          (axes[1], layout, "optimized")):
        draw_layout_topview(lay, context, ax, title=f"{name} — defect region")
        ax.set_xlim(-3.5, 3.5)
    axes[1].plot([0], [0], "r*", ms=10)
    p2 = REP_DIR / "fig_design_zoom.png"
    fig.savefig(p2, dpi=200); plt.close(fig)
    return p1, p2


def fig_params(rec):
    layout, _ = layout_from_entry(rec[-1])
    base = baseline_layout()
    idx = np.arange(len(layout["a"]))

    fig, axes = plt.subplots(3, 1, figsize=(9, 7.5), tight_layout=True,
                             sharex=True)
    panels = [("a", "lattice constant a$_i$ (µm)"),
              ("sx", "hole x-scale s$_x$ (µm)"),
              ("sy", "hole y-scale s$_y$ (µm)")]
    for ax, (key, label) in zip(axes, panels):
        ax.plot(idx, np.asarray(base[key]), "o--", ms=3, color="C3",
                label="baseline", alpha=0.8)
        ax.plot(idx, np.asarray(layout[key]), "o-", ms=3, color="C0",
                label="optimized")
        ax.set_ylabel(label, fontsize=9)
        ax.legend(fontsize=8)
    axes[0].set_title("Per-hole parameters (sections: 5 taper | 10 mirror | "
                      "40 defect | 30 mirror | 1 taper; output port at left)",
                      fontsize=10)
    axes[-1].set_xlabel("hole index (left → right)")
    for ax in axes:
        for b in (5, 15, 55, 85):
            ax.axvline(b - 0.5, color="gray", lw=0.6, alpha=0.5)
    p = REP_DIR / "fig_params.png"
    fig.savefig(p, dpi=200); plt.close(fig)
    return p


def fig_shapes(rec):
    layout, _ = layout_from_entry(rec[-1])
    base = baseline_layout()
    picks = [(35, "defect-center cell"), (70, "back-mirror cell"),
             (10, "output-side mirror cell")]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), tight_layout=True)
    for ax, (i, name) in zip(axes, picks):
        vb = np.asarray(base["vertices"][i])
        vo = np.asarray(layout["vertices"][i])
        ax.plot(*np.vstack([vb, vb[:1]]).T * 1000, "--", color="C3",
                label="baseline ellipse")
        ax.plot(*np.vstack([vo, vo[:1]]).T * 1000, "-", color="C0",
                label="optimized spline")
        ax.set_title(f"hole {i}: {name}", fontsize=9)
        ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
        ax.set_aspect("equal"); ax.legend(fontsize=7)
    p = REP_DIR / "fig_shapes.png"
    fig.savefig(p, dpi=200); plt.close(fig)
    return p


def fig_fields():
    """Re-render fields from the final verification sim (horizontal layout)."""
    import tidy3d as td
    sd = td.SimulationData.from_file(str(RUN_DIR / "verification" /
                                         "final_design.hdf5"))
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), tight_layout=True)
    specs = [("Field_Profile_Monitor_z", "Ey", "Re(E$_y$), z = 0 (beam midplane)", "RdBu"),
             ("Field_Profile_Monitor_z", "abs", "|E|, z = 0", "magma"),
             ("Field_Profile_Monitor_y", "abs", "|E|, y = 0 (vertical cut; atom sits 200–300 nm above the top surface)", "magma")]
    for ax, (mon, comp, title, cmap) in zip(axes, specs):
        data = sd[mon]
        if comp == "abs":
            arr = np.sqrt(sum(np.abs(getattr(data, c).isel(t=-1)) ** 2
                              for c in ("Ex", "Ey", "Ez"))).squeeze()
        else:
            arr = np.real(getattr(data, comp).isel(t=-1)).squeeze()
        ydim = "y" if mon.endswith("_z") else "z"
        im = ax.pcolormesh(arr.x, arr[ydim], arr.T, cmap=cmap, shading="auto")
        plt.colorbar(im, ax=ax, pad=0.01)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x (µm)"); ax.set_ylabel(f"{ydim} (µm)")
        ax.set_xlim(-18, 18); ax.set_aspect("auto")
    p = REP_DIR / "fig_fields.png"
    fig.savefig(p, dpi=200); plt.close(fig)
    return p


def fig_evolution(rec):
    iters = [0, len(rec) // 2, len(rec) - 1]
    fig, axes = plt.subplots(len(iters), 1, figsize=(12, 6), tight_layout=True)
    for ax, j in zip(axes, iters):
        layout, context = layout_from_entry(rec[j])
        e = rec[j]
        draw_layout_topview(layout, context, ax,
                            title=f"iteration {e['params']['iteration']} — "
                                  f"loss {e['loss']:.2f}")
    p = REP_DIR / "fig_evolution.png"
    fig.savefig(p, dpi=200); plt.close(fig)
    return p


def band_edges(bands: dict, name: str):
    """(band-1 X-point edge, band-2 X-point edge) in Hz, or (None, None)."""
    r = bands[name]
    fx = sorted(f for f in r["freqs"][-1])
    if not fx:
        return None, None
    e1 = fx[0]
    e2 = next((f for f in fx[1:] if f - e1 > 5e12), None)
    return e1, e2


# ──────────────────────────────────────────────────────────────────────────
# PDF assembly
# ──────────────────────────────────────────────────────────────────────────

class Report:
    def __init__(self):
        from fpdf import FPDF
        self.pdf = FPDF(format="A4")
        self.pdf.set_auto_page_break(auto=True, margin=18)
        self.pdf.add_font("dejavu", "", str(FONT_DIR / "DejaVuSans.ttf"))
        self.pdf.add_font("dejavu", "B", str(FONT_DIR / "DejaVuSans-Bold.ttf"))
        # no oblique TTF on this system; serif stands in for emphasis
        self.pdf.add_font("dejavu", "I", str(FONT_DIR / "DejaVuSerif.ttf"))
        self.W = self.pdf.epw

    def h1(self, text):
        self.pdf.set_font("dejavu", "B", 15)
        self.pdf.multi_cell(self.W, 8, text)
        self.pdf.ln(1)

    def h2(self, text):
        self.pdf.ln(2)
        self.pdf.set_font("dejavu", "B", 12)
        self.pdf.multi_cell(self.W, 6.5, text)
        self.pdf.ln(1)

    def p(self, text, size=9.5, style=""):
        self.pdf.set_font("dejavu", style, size)
        self.pdf.multi_cell(self.W, 4.8, text)
        self.pdf.ln(1.5)

    def img(self, path, w=None, caption=None):
        self.pdf.image(str(path), w=w or self.W)
        if caption:
            self.pdf.set_font("dejavu", "I", 8.5)
            self.pdf.multi_cell(self.W, 4.2, caption)
        self.pdf.ln(2)

    def table(self, headers, rows):
        self.pdf.set_font("dejavu", "", 9)
        with self.pdf.table(text_align="CENTER", line_height=5.5,
                            col_widths=[34, 22, 22, 22]) as table:
            hr = table.row()
            for h in headers:
                hr.cell(h)
            for row in rows:
                r = table.row()
                for c in row:
                    r.cell(str(c))
        self.pdf.ln(2)

    def table_wide(self, headers, rows):
        """Table with a label column plus N equal data columns."""
        self.pdf.set_font("dejavu", "", 8.5)
        n_data = len(headers) - 1
        widths = [30] + [18] * n_data
        with self.pdf.table(text_align="CENTER", line_height=5.2,
                            col_widths=widths) as table:
            hr = table.row()
            for h in headers:
                hr.cell(h)
            for row in rows:
                r = table.row()
                for c in row:
                    r.cell(str(c))
        self.pdf.ln(2)


def build_pdf(rec, baseline, bands):
    final = rec[-1]["perf"]["diagnostic"]
    bp = baseline["baseline_perf"]
    total_time = sum(e["time"]["duration_s"] for e in rec)

    r = Report()
    pdf = r.pdf
    pdf.add_page()

    # ── title & summary ───────────────────────────────────────────────
    r.h1("Inverse design of a suspended SiN nanobeam cavity for a "
         "⁸⁷Rb–waveguide photon interface")
    r.p("Repository: atomTrap — automated inverse-design workflow "
        "(inverse_design/ package, Tidy3D autograd adjoint optimization). "
        "Run: invDesResults/full, 42 recorded iterations, "
        f"{total_time/3600:.2f} h optimization time, ≈67 FlexCredits total.",
        style="I")

    r.h2("Summary")
    r.p("The hand-tuned baseline (design_1 from cavities_780.ipynb) was converted into a fully "
        "differentiable parameterization — 86 per-cell lattice constants, 172 per-hole scale "
        "factors, and 20 closed-spline shape control points (278 parameters) — and optimized "
        "against a frequency-domain surrogate of the physical photon-interface efficiency "
        "η = (κ_wg/κ_tot) · C/(C+1) · η_fiber, with resonance targeting at the ⁸⁷Rb D2 line "
        "(384.23 THz) and fabrication constraints (50 nm hole-hole gaps, 70 nm side rails, "
        "40 nm curvature radius). Every reported physical number below comes from the repo's "
        "pre-existing time-domain verification pipeline (ResonanceFinder Q, per-face directional "
        "Q, mode volume), applied identically to baseline and optimized designs.")
    r.p("Headline result: the verified end-to-end collection efficiency at a realistic atom "
        "position (250 nm above the beam surface) improved 3.4×, from η = 0.083 to η = 0.278, "
        "while the resonance moved from 1.01 THz off the Rb D2 target to 0.29 THz — within "
        "thermal-tuning range. Cooperativity at the atom doubled (C = 21 → 45).")

    # ── performance table ─────────────────────────────────────────────
    r.h2("1. Verified performance vs. baseline")
    rows = [
        ["f_res (THz) [target 384.23]", f"{bp['f_res_Hz']/1e12:.2f}",
         f"{final['f_res_Hz']/1e12:.2f}", "Δ: 1.01→0.29 THz"],
        ["η (atom @250 nm)", f"{bp['eta_atom']:.3f}", f"{final['eta_atom']:.3f}",
         f"{final['eta_atom']/bp['eta_atom']:.2f}×"],
        ["C (atom @250 nm)", f"{bp['C_atom']:.1f}", f"{final['C_atom']:.1f}",
         f"{final['C_atom']/bp['C_atom']:.2f}×"],
        ["β = κ_wg/κ_tot", f"{bp['beta_wg']:.3f}", f"{final['beta_wg']:.3f}",
         f"{final['beta_wg']/bp['beta_wg']:.2f}×"],
        ["Q (loaded)", f"{bp['Q']:.0f}", f"{final['Q']:.0f}",
         f"{final['Q']/bp['Q']:.2f}×"],
        ["κ_tot/2π (GHz)", f"{bp['kappa_tot_2pi_GHz']:.0f}",
         f"{final['kappa_tot_2pi_GHz']:.0f}", "—"],
        ["g/2π @250 nm (GHz)", f"{bp['g_2pi_Hz_at_250nm']/1e9:.2f}",
         f"{final['g_2pi_Hz_at_250nm']/1e9:.2f}",
         f"{final['g_2pi_Hz_at_250nm']/bp['g_2pi_Hz_at_250nm']:.2f}×"],
        ["V_mode ((λ/n)³, legacy conv.)", f"{bp['Vmode_lambda_n3']:.2f}",
         f"{final['Vmode_lambda_n3']:.2f}", "—"],
        ["Purcell (3Q/4π²V)", f"{bp['purcell']:.0f}", f"{final['purcell']:.0f}",
         f"{final['purcell']/bp['purcell']:.2f}×"],
    ]
    r.table(["metric", "baseline", "optimized", "change"], rows)
    r.p("Note on the baseline: the notebook-recorded Q = 1.33×10⁶ for design_1 is not "
        "reproducible from the notebook's stored parameters (its 0.5 µm left-mirror lattice has "
        "a/λ ≈ 0.64 at 780 nm, far above the first TE gap — classic notebook cell-edit drift). "
        "The stored parameter set is the only reproducible 'current best design'; it was "
        "re-evaluated with the identical pipeline used for the optimized design.", size=8.5)

    # ── design ────────────────────────────────────────────────────────
    pdf.add_page()
    r.h2("2. The optimized design")
    r.img(REP_DIR / "fig_design_full.png",
          caption="Fig. 1 — Top view (z = 0): baseline vs. optimized. Output waveguide at left "
                  "(5-hole taper + 10 mirror cells); 40-cell defect; 30 back-mirror cells at right.")
    r.img(REP_DIR / "fig_design_zoom.png",
          caption="Fig. 2 — Defect region (red star: dipole/atom x-position at the central "
                  "dielectric antinode).")

    pdf.add_page()
    r.img(REP_DIR / "fig_params.png", w=r.W * 0.92,
          caption="Fig. 3 — Per-hole parameters, baseline (red dashed) vs. optimized (blue). The "
                  "optimizer kept the defect-region lattice near 0.29 µm but reshaped the "
                  "left-mirror region (originally a = 0.5 µm — not a working mirror) into a "
                  "graded structure, and softened the size discontinuities at section boundaries "
                  "into smooth tapers (gentle confinement).")
    r.img(REP_DIR / "fig_shapes.png", w=r.W * 0.92,
          caption="Fig. 4 — Closed-spline hole boundaries vs. baseline ellipses for three "
                  "representative cells. Shape deformations are modest (control radii within "
                  "±3% of circular): at these learning rates the spacing/size schedule, not the "
                  "hole shape, was the dominant lever.")

    # ── bands ─────────────────────────────────────────────────────────
    pdf.add_page()
    r.h2("3. Band structure of the constituent unit cells")
    eo1, _ = band_edges(bands, "opt_defect")
    em1, em2 = band_edges(bands, "opt_mirror")
    eb1, eb2 = band_edges(bands, "base_mirror")
    r.img(REP_DIR / "bands.png",
          caption="Fig. 5 — TE-like band structures of six constituent unit cells (Tidy3D "
                  "Bloch unit-cell simulations of the suspended beam, dipole excitation + "
                  "resonance analysis; MPB is unavailable in this environment). Gray: light "
                  "cone; red dashed: 384.23 THz Rb D2 target.")
    r.p("The cavity works exactly as the quasipotential picture (Knall et al., PRL 129, 053603 "
        "(2022)) prescribes. In the defect cell the dielectric band edge at the zone boundary "
        f"sits at {eo1/1e12:.0f} THz — the 383.9 THz resonance lies below it, in the defect's "
        "propagating dielectric band. In the back-mirror cell the gap spans "
        f"{em1/1e12:.0f}–{em2/1e12:.0f} THz, so the resonance sits near mid-gap where mirror "
        "strength is maximal — light is evanescent there and the back side reflects well "
        f"(verified Q₊ₓ = 2.7×10⁵). The baseline's equivalent cells are nearly identical "
        f"(defect edge {band_edges(bands, 'base_defect')[0]/1e12:.0f} THz, back-mirror gap "
        f"{eb1/1e12:.0f}–{eb2/1e12:.0f} THz): the unit-cell photonics was never the baseline's "
        "problem.")
    r.p("The output-side cells are the story. With a ≈ 0.5 µm, all guided bands at the zone "
        "boundary lie below ≈ 295 THz and the light line at X is c/2a ≈ 300 THz: at 384 THz "
        "this section has no band gap and no guided modes — it is not a mirror but a radiative "
        "scatterer, in BOTH designs. This is why the baseline lost 91% of its photons to "
        "scattering (β = 0.087, Q_y ≈ 940 per face). The optimizer, rather than rebuilding "
        "those cells, constructed a smooth graded barrier from the defect-adjacent cells "
        "(the lattice descent 0.50 → 0.29 µm visible in Fig. 3, indices 15–27) — the "
        "gentle-confinement mechanism of Quan & Lončar (2011) / Tanaka et al. (2008): a "
        "gradually rising local band edge confines the mode with minimal Fourier content "
        "inside the light cone. That raised β to 0.287 and Q to 1778; the residual a = 0.5 µm "
        "segment still radiates and remains the dominant loss channel.")

    # ── fields ────────────────────────────────────────────────────────
    pdf.add_page()
    r.h2("4. Cavity mode fields")
    r.img(REP_DIR / "fig_fields.png",
          caption="Fig. 6 — Verified cavity mode of the optimized design (final full-quality "
                  "time-domain simulation, last time step). Top: Re(Ey) at z = 0 showing the "
                  "Bloch-wave standing pattern under a localized envelope; middle: |E| at z = 0; "
                  "bottom: |E| in the vertical cut — the evanescent tail above the top surface "
                  "is what couples to the tweezer-trapped atom at 200–300 nm.")
    r.p(f"At the atom positions the verified coupling is g/2π = "
        f"{final['g_2pi_Hz_at_200nm']/1e9:.2f} / {final['g_2pi_Hz_at_250nm']/1e9:.2f} / "
        f"{final['g_2pi_Hz_at_300nm']/1e9:.2f} GHz at 200/250/300 nm — the ~1.9× decay over "
        "100 nm quantifies the design's sensitivity to the tweezer trap distance (set by beam "
        "thickness via the retro-reflection phase; Thompson et al. 2013).")

    # ── optimization trajectory ───────────────────────────────────────
    pdf.add_page()
    r.h2("5. Optimization trajectory")
    r.img(RUN_DIR / "loss_history.png", w=r.W * 0.8,
          caption="Fig. 7 — Loss vs. iteration. The spike at iteration 24 is the resonance-"
                  "targeting penalty unsaturating when the monitor window was widened (see text).")
    r.img(REP_DIR / "fig_evolution.png", w=r.W * 0.95,
          caption="Fig. 8 — Structure evolution (full animation: invDesResults/full/"
                  "structure_evolution.gif).")
    pdf.add_page()
    r.img(RUN_DIR / "perf_history.png", w=r.W * 0.88,
          caption="Fig. 9 — Verified physical parameters vs. iteration; dashed red: baseline.")
    r.p("Two-stage schedule: Stage A (iterations 2–23) optimized spacings and hole scales with "
        "circular spline control points; Stage B (24–43) unlocked the spline shapes. During "
        "Stage A the η term walked the resonance 11 THz below target — with a narrow 2 THz "
        "monitor window the softmax frequency estimator saturates at the window edge and the "
        "resonance penalty loses its gradient. Widening the window to 6 THz and anchoring it to "
        "cover both the verified mode and the target restored the penalty: iterations 24–31 "
        "brought the resonance back to 384.2 THz while η continued to rise — the optimizer found "
        "designs that are simultaneously on-target and better-coupled.")

    # ── literature context ────────────────────────────────────────────
    pdf.add_page()
    r.h2("6. Physical impact in the context of the literature")
    g, k_tot = final["g_2pi_Hz_at_250nm"], final["kappa_tot_2pi_GHz"] * 1e9
    C, eta, beta = final["C_atom"], final["eta_atom"], final["beta_wg"]
    err_o, err_b = np.log(C) / C, np.log(bp["C_atom"]) / bp["C_atom"]
    k_s = k_tot * (1 - beta)
    gamma = 6.0666e6
    k_opt = np.sqrt(k_s * (4 * g ** 2 + k_s * gamma) / gamma)
    r.p("Atom–photon entanglement fidelity. For time-bin atom–photon entanglement through a "
        "nanophotonic cavity the infidelity scales as ln C/C (Menon et al., NJP 22, 073033 "
        f"(2020)). The improvement C = 21 → 45 reduces this error bound from "
        f"≈{err_b*100:.0f}% to ≈{err_o*100:.1f}% (F ≈ 0.9 requires C ≳ 10–15; demonstrated "
        "single-atom records are C ≈ 27–71, Suleymanzade & Dimitrova 2025 review). The optimized "
        "design's cooperativity at a realistic 250 nm trap position is therefore in the regime "
        "of the best demonstrated atom–nanophotonic interfaces, whereas the baseline sat at the "
        "threshold.")
    r.p("Remote-entanglement rate. The collection efficiency η enters two-node entanglement "
        "rates quadratically (both photons must be collected): η = 0.083 → 0.278 is a "
        f"{(final['eta_atom']/bp['eta_atom'])**2:.0f}× speed-up of heralded Bell-pair rates — "
        "the difference between sub-kHz and multi-kHz operation at typical attempt rates, "
        "directly addressing the >10 kHz remote-entanglement target highlighted in the "
        "Suleymanzade & Dimitrova review.")
    r.p("Bad-cavity regime. The hierarchy κ/2π = 216 GHz ≫ g/2π = 3.8 GHz ≫ γ/2π = 6.1 MHz "
        "places the device deep in the fast/bad-cavity Purcell regime that the Reiserer & Rempe "
        "(RMP 2015) and Knall et al. (PRL 2022) analyses favor for broadband, fast photon "
        "extraction: protocol bandwidth is set by κ, and the Duan–Kimble reflection gate "
        "mechanism is robust to the exact g in this regime.")
    r.p("Where the design still falls short. (i) Outcoupling: β = κ_wg/κ_tot = 0.287 — better "
        "than the 10–40% measured in the closest predecessor platform (Menon thesis 2025) but "
        "below its >60% goal, and far below the Knall-optimal overcoupling "
        f"(κ_wg,opt/2π = √(κ_s(4g²+κ_sγ)/γ)/2π ≈ {k_opt/1e12:.1f} THz ≫ κ_s/2π = "
        f"{k_s/1e9:.0f} GHz, i.e. the source-efficiency optimum wants β → ≈0.9): scattering "
        "still carries ~71% of the photons. (ii) Loaded Q = 1778 is orders below the Q ~ 10⁴–10⁵ "
        "that gentle-confinement designs reach (Quan & Lončar 2011; Bazin et al. 2014); raising "
        "intrinsic Q would raise C and β simultaneously. Both point the same way: this seed "
        "design family (the reproducible but weak design_1) limits the achievable basin — the "
        "machinery, which is seed-agnostic, should next be run from a band-structure-designed "
        "seed (β was still rising when the iteration budget ended; resume from "
        "invDesResults/full/checkpoints/latest.pkl).", size=9.5)
    r.p("Caveats on absolute numbers. g and V are quoted via the repo's legacy conventions "
        "(time-averaged energy-density ratios from the verification sims); the legacy V_mode "
        "includes a symmetry double-count kept for continuity, and g is evaluated on the beam "
        "axis directly above the defect. Absolute values should be treated as ~factor-2 "
        "estimates; the baseline-to-optimized ratios are computed identically on both designs "
        "and are robust.", size=8.5)

    # ── reproducibility ───────────────────────────────────────────────
    r.h2("7. Reproducibility")
    r.p("Record: invDesResults/full/record.json — 42 entries with keys 'params' (full "
        "reconstruction state), 'loss', 'time', 'perf'; geometry round-trip verified. "
        "Checkpoints: invDesResults/full/checkpoints/ (per-iteration resume state). "
        "Commands (tidyEnv, repo root): smoke test "
        "'python -m inverse_design.run_optimization --mode smoke'; full run / extension "
        "'python -m inverse_design.run_optimization --mode full --run-name full --resume'; "
        "band diagrams 'python -m inverse_design.bands'; this report "
        "'python -m inverse_design.make_report'. Unit tests: 'pytest tests/' (19 pass). "
        "GDS of the optimized design: invDesResults/full/final_design.gds.", size=9)

    pdf.output(str(PDF_PATH))
    return PDF_PATH


def main():
    REP_DIR.mkdir(parents=True, exist_ok=True)
    rec = rm.load_record(RUN_DIR)
    with open(RUN_DIR / "baseline.json") as f:
        baseline = json.load(f)
    with open(REP_DIR / "bands.json") as f:
        bands = json.load(f)

    fig_design(rec)
    fig_params(rec)
    fig_shapes(rec)
    fig_fields()
    fig_evolution(rec)
    path = build_pdf(rec, baseline, bands)
    print("report ->", path)


if __name__ == "__main__":
    main()
