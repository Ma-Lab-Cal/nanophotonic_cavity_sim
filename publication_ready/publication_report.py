"""Assemble a single review PDF from the figure PNGs + datasets.

Convenience document (publication_ready/Publication_Report.pdf) collecting the
eight main figures, the supplements, the comparison table, the acceptance-gate
verdict, and the convention summary. Text is ASCII-safe (greek spelled out) so
no special fonts are needed; the figures themselves carry the symbols.

    python -m publication_ready.publication_report
"""

from __future__ import annotations

from fpdf import FPDF

from publication_ready import campaign, paths
from publication_ready.lib import datasets

MAIN_FIGS = [
    ("Fig1_device", "Fig. 1 — Device geometry and per-hole parameter schedule "
     "(seed vs final)."),
    ("Fig2_bands", "Fig. 2 — Bloch band structure of the mirror and defect "
     "cells; band gap, light line, Rb D2 target, and final f_res."),
    ("Fig3_resonance_loss", "Fig. 3 — Resonance fit and the waveguide-extraction "
     "loss budget (kappa_wg dominant; residual = 1 - beta)."),
    ("Fig4_fields", "Fig. 4 — Mode localization and field at the atom position."),
    ("Fig5_tuning", "Fig. 5 — Tuning sweeps (global / defect / output) and the "
     "overcoupling Pareto map."),
    ("Fig6_position", "Fig. 6 — Atom-position robustness of eta and C."),
    ("Fig7_fabrication", "Fig. 7 — Fabrication-tolerance robustness (corner "
     "grid), unretuned and retuned."),
    ("Fig8_comparison", "Fig. 8 — Five-design comparison in one metric "
     "convention."),
]
SUPP_FIGS = [
    ("SuppA_convergence", "Supplement A — Numerical convergence (mesh / PML / "
     "run-time)."),
    ("SuppB_mode_isolation", "Supplement B — Mode isolation."),
]


class _PDF(FPDF):
    def header(self):
        pass

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 7)
        self.cell(0, 8, "SiN nanobeam Rb cavity - publication package - "
                  "EM-only scalar metrics", align="C")


_SUBS = {"Δ": "d", "β": "beta", "η": "eta", "κ": "kappa", "λ": "lambda",
         "γ": "gamma", "²": "^2", "³": "^3", "×": "x", "≤": "<=", "≥": ">=",
         "→": "->", "−": "-", "·": ".", "±": "+/-", "μ": "u", "—": "-", "–": "-"}


def _ascii(s: str) -> str:
    for k, v in _SUBS.items():
        s = s.replace(k, v)
    return s.encode("ascii", "replace").decode("ascii")


def _png(name):
    for d in (paths.FIG_MAIN, paths.FIG_SUPP):
        p = d / f"{name}.png"
        if p.exists():
            return p
    return None


def build():
    pdf = _PDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(True, margin=15)
    W = 180

    # ── title ───────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.multi_cell(0, 9, "Publication package: SiN nanobeam cavity\n"
                   "tuned to the Rb D2 line")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)

    nv_path = paths.DATA_DIR / "nominal_verify.json"
    if nv_path.exists():
        nv = datasets.read_json(nv_path)
        perf = nv["perf"]
        gate = nv["gate"]
        det = nv["detuning_GHz"]
        summary = (
            f"Final trimmed design: f_res = {perf['f_res_Hz']/1e12:.4f} THz "
            f"({det:+.1f} GHz from Rb D2), Q = {perf['Q']:.0f}, "
            f"kappa/2pi = {perf['kappa_tot_2pi_GHz']:.1f} GHz, "
            f"beta = {perf['beta_wg']:.3f}, C = {perf['C_atom']:.1f}, "
            f"eta = {perf['eta_atom']:.3f}, V = {perf['Vmode_lambda_n3']:.3f} "
            f"(lambda/n)^3.\n\n"
            f"Acceptance gate: {'PASSED' if gate['passed'] else 'NOT PASSED'}"
            f"  (tolerance |df| <= {gate['tol_hz']/1e9:.1f} GHz). "
            + ("" if gate["passed"] else " Open items: " + "; ".join(gate["reasons"]))
        )
    else:
        summary = ("Final verification not yet run (run nominal_verify.py). "
                   "This report shows whatever figures are available.")
    pdf.multi_cell(0, 5, _ascii(summary))
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 4.5,
                   "Scope: EM-derived scalar proxies (g, C, eta, beta) from "
                   "classical FDTD; not a cavity-QED simulation. No gate/readout/"
                   "entanglement-fidelity claims. See limitations.md.")

    # ── comparison table ─────────────────────────────────────────────────────
    cmp_path = paths.TABLES_DIR / "comparison_table.csv"
    if cmp_path.exists():
        rows = datasets.read_csv(cmp_path)
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Five-design comparison (one convention)", ln=1)
        pdf.set_font("Helvetica", "B", 8)
        cols = [("design", 42), ("f_res_THz", 24), ("detuning_GHz", 24),
                ("eta_atom", 18), ("beta_wg", 16), ("C_atom", 18), ("Q", 22)]
        for name, w in cols:
            pdf.cell(w, 6, name, border=1)
        pdf.ln()
        pdf.set_font("Helvetica", "", 8)
        for r in rows:
            for name, w in cols:
                v = r.get(name, "")
                if isinstance(v, float):
                    v = (f"{v:.3f}" if name in ("eta_atom", "beta_wg")
                         else f"{v:.1f}" if name in ("detuning_GHz", "C_atom",
                                                     "Q")
                         else f"{v:.4f}" if name == "f_res_THz" else f"{v}")
                pdf.cell(w, 6, str(v), border=1)
            pdf.ln()

    # ── figures ──────────────────────────────────────────────────────────────
    for name, caption in MAIN_FIGS + SUPP_FIGS:
        png = _png(name)
        if png is None:
            continue
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 11)
        pdf.multi_cell(0, 6, _ascii(caption.split("—")[0].strip()))
        pdf.image(str(png), w=W)
        pdf.ln(1)
        pdf.set_font("Helvetica", "I", 9)
        pdf.multi_cell(0, 4.5, _ascii(caption))

    out = paths.PUB_ROOT / "Publication_Report.pdf"
    pdf.output(str(out))
    print(f"[report] wrote {out}")
    return out


if __name__ == "__main__":
    build()
