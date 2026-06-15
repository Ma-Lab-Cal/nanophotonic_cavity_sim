"""Publication figure plotters (data-only; never touch a live sim).

Each function takes already-loaded data (dicts/arrays from the saved csv/json/npz
datasets) and writes one PDF. The three plotters flagged as missing from
``inverse_design.visualization`` (per-hole parameter-trace overlay, atom-position
heatmaps, multi-design bars) live here; the rest compose matplotlib directly.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from inverse_design import config as cfg
from inverse_design.visualization import draw_layout_topview

F_D2_THZ = cfg.F_TARGET / 1e12

DESIGN_ORDER = ["design_1", "invDes(design_1)", "seed", "invDes(seed)", "final"]
DESIGN_COLORS = {
    "design_1": "#888888", "invDes(design_1)": "#1f77b4", "seed": "#2ca02c",
    "invDes(seed)": "#ff7f0e", "final": "#d62728",
}


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Fig 1: device + per-hole parameter traces (seed vs final) ──────────────

def fig_device_and_traces(final_layout, final_ctx, seed_layout, out_path):
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(4, 1, height_ratios=[2.2, 1, 1, 1], hspace=0.45)
    ax0 = fig.add_subplot(gs[0])
    draw_layout_topview(final_layout, final_ctx, ax0,
                        title="Final trimmed design — top view (z = 0)")
    # mark the atom / output port
    ax0.annotate("output (−x)", xy=(final_layout["positions"][0], 0),
                 xytext=(final_layout["positions"][0], final_ctx["width"]),
                 fontsize=8, ha="center",
                 arrowprops=dict(arrowstyle="->", color="C3"))
    ax0.scatter([0], [0], marker="*", s=120, color="red", zorder=5,
                label="atom @ x=0")
    ax0.legend(loc="upper right", fontsize=8)

    nf = len(final_layout["a"])
    idx = np.arange(nf)
    panels = [("a", "lattice a (nm)", 1e3),
              ("sx", "hole x-diam sₓ (nm)", 1e3),
              ("sy", "hole y-diam s_y (nm)", 1e3)]
    for r, (key, label, sc) in enumerate(panels):
        ax = fig.add_subplot(gs[r + 1])
        fv = np.asarray(final_layout[key]) * sc
        ax.plot(idx, fv, "o-", ms=3, color="C3", label="final")
        if seed_layout is not None and len(seed_layout[key]) == nf:
            sv = np.asarray(seed_layout[key]) * sc
            ax.plot(idx, sv, "s--", ms=2.5, color="C2", alpha=0.7, label="seed")
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.25)
        if r == 0:
            ax.legend(fontsize=8, ncol=2, loc="upper center")
        if r == 2:
            ax.set_xlabel("hole index (−x output → +x back)")
    fig.suptitle("Fig. 1 — Device geometry and per-hole parameter schedule",
                 fontsize=12, y=0.93)
    return _save(fig, out_path)


# ── Fig 2: Bloch bands (mirror + defect cells) ─────────────────────────────

def _bandgap_edges(cell):
    """Crude gap = between the two lowest X-point bands of a mirror cell."""
    xfreqs = sorted(np.asarray(cell["freqs"][-1], dtype=float) / 1e12)
    if len(xfreqs) >= 2:
        return xfreqs[0], xfreqs[1]
    return None, None


def fig_bands(bands, out_path, f_res_thz=None,
              cell_titles=None):
    names = list(bands)
    fig, axes = plt.subplots(1, len(names), figsize=(4.4 * len(names), 4.8),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, name in zip(axes, names):
        r = bands[name]
        a = r["a"]
        for k, fr in zip(r["ks"], r["freqs"]):
            ax.scatter([k] * len(fr), np.asarray(fr) / 1e12, s=16, color="C0",
                       alpha=0.85, zorder=3)
        kk = np.linspace(0, 0.5, 60)
        ll = kk * cfg.C0_UM / a * 1e-12
        ax.plot(kk, ll, "k-", lw=1, alpha=0.5, label="light line")
        ax.fill_between(kk, ll, 480, color="gray", alpha=0.12)
        lo, hi = _bandgap_edges(r)
        if lo and hi and "mir" in name.lower() or (lo and hi and "mirror" in name.lower()):
            ax.axhspan(lo, hi, color="C1", alpha=0.12, label="band gap")
        ax.axhline(F_D2_THZ, color="C3", ls="--", lw=1.3, label="Rb D2 (384.23)")
        if f_res_thz is not None:
            ax.axhline(f_res_thz, color="C2", ls=":", lw=1.3,
                       label=f"cavity f_res ({f_res_thz:.1f})")
        ax.set_xlim(0, 0.5)
        ax.set_ylim(300, 480)
        ax.set_xlabel(r"$k_x$ ($2\pi/a$)")
        ax.set_title((cell_titles or {}).get(name, name) +
                     f"\na = {a*1000:.0f} nm", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("frequency (THz)")
            ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Fig. 2 — Bloch band structure of the mirror and defect cells",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, out_path)


# ── Fig 3: resonance spectrum + loss budget ────────────────────────────────

def fig_resonance_loss(nominal, out_path):
    perf = nominal["perf"]
    spec = nominal["resonance_spectrum"]
    budget = nominal["loss_budget"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    f = np.asarray(spec["freq_Hz"]) / 1e12
    q = np.asarray(spec["Q"])
    ax.stem(f, q, basefmt=" ")
    ax.axvline(F_D2_THZ, color="C3", ls="--", label="Rb D2")
    ax.axvline(perf["f_res_Hz"] / 1e12, color="C2", ls=":",
               label=f"f_res ({perf['f_res_Hz']/1e12:.4f} THz)")
    ax.set_xlabel("frequency (THz)")
    ax.set_ylabel("fitted Q")
    ax.set_title(f"Resonance fit  (Q={perf['Q']:.0f}, "
                 f"κ/2π={perf['kappa_tot_2pi_GHz']:.1f} GHz, "
                 f"fit err={budget['fit_error']:.1e})")
    ax.legend(fontsize=8)

    ax = axes[1]
    faces = budget["kappa_per_face_2pi_GHz"]
    out_face = budget["output_face"]
    labels, vals, colors = [], [], []
    order = [out_face] + [k for k in ("+y", "-y", "+z", "-z", "+x")
                          if k in faces and k != out_face]
    for k in order:
        labels.append("κ_wg (−x)" if k == out_face else f"κ_{k}")
        vals.append(faces[k])
        colors.append("C3" if k == out_face else "C0")
    ax.bar(labels, vals, color=colors)
    ax.set_ylabel("κ / 2π (GHz)")
    ax.set_title(f"Loss budget  (β = κ_wg/κ_tot = {budget['beta_wg']:.3f}, "
                 f"residual = {budget['residual_fraction']*100:.0f}%)")
    ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Fig. 3 — Resonance and waveguide-extraction loss budget",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, out_path)


# ── Fig 4: field localization ──────────────────────────────────────────────

def fig_fields(fields, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 6.5))

    def _imshow(ax, tag, comp, label, cmap):
        ak = f"{tag}_{comp}"
        if ak not in fields:
            ax.set_visible(False)
            return
        dims = [str(x) for x in fields[f"{tag}_dims"]]
        A = np.asarray(fields[ak])
        c0 = np.asarray(fields[f"{tag}_{dims[0]}"])
        c1 = np.asarray(fields[f"{tag}_{dims[1]}"])
        im = ax.pcolormesh(c0, c1, A.T, cmap=cmap, shading="auto",
                           rasterized=True)
        ax.set_xlabel(f"{dims[0]} (µm)")
        ax.set_ylabel(f"{dims[1]} (µm)")
        ax.set_title(label)
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.025)

    _imshow(axes[0, 0], "midplane_z", "Eabs", "|E| — beam midplane (z=0)", "magma")
    _imshow(axes[0, 1], "midplane_z", "Ey", "Re(E_y) — beam midplane (z=0)", "RdBu")
    _imshow(axes[1, 0], "vertical_x", "Eabs", "|E| — atom plane (x=0, yz)", "magma")
    _imshow(axes[1, 1], "vertical_y", "Eabs", "|E| — longitudinal (y=0, xz)", "magma")
    # mark atom height on the vertical plane
    zatom = float(fields["atom_height_um"])
    axes[1, 0].axhline(zatom, color="cyan", ls="--", lw=1)
    axes[1, 0].annotate("atom (250 nm)", xy=(0, zatom), color="cyan", fontsize=8)
    fig.suptitle("Fig. 4 — Mode localization and field at the atom position",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, out_path)


# ── Fig 5: tuning / Pareto map ─────────────────────────────────────────────

def _col(rows, key):
    return np.array([r[key] for r in rows], dtype=float)


def fig_tuning(global_rows, defect_rows, output_rows, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (a) detuning vs global scale
    ax = axes[0, 0]
    gs = _col(global_rows, "value")
    ax.plot(gs, _col(global_rows, "detuning_GHz"), "o-", color="C0")
    ax.axhline(0, color="C3", ls="--", label="Rb D2")
    for r in global_rows:
        if r.get("design_marker") == "final":
            ax.scatter([r["value"]], [r["detuning_GHz"]], marker="*", s=180,
                       color="C3", zorder=5, label="selected final")
    ax.set_xlabel("global in-plane scale s")
    ax.set_ylabel("detuning from D2 (GHz)")
    ax.set_title("(a) Global-scale resonance trim")
    ax.legend(fontsize=8)

    # (b) eta & beta vs global scale
    ax = axes[0, 1]
    ax.plot(gs, _col(global_rows, "eta_atom"), "o-", color="C0", label="η")
    ax.plot(gs, _col(global_rows, "beta_wg"), "s--", color="C1", label="β")
    ax.set_xlabel("global in-plane scale s")
    ax.set_ylabel("η, β")
    ax.set_title("(b) Interface metrics vs trim (≈invariant)")
    ax.legend(fontsize=8)

    # (c) defect-only axis
    ax = axes[1, 0]
    if defect_rows:
        dv = _col(defect_rows, "value")
        ax.plot(dv, _col(defect_rows, "detuning_GHz"), "o-", color="C4",
                label="detuning")
        ax2 = ax.twinx()
        ax2.plot(dv, _col(defect_rows, "eta_atom"), "s--", color="C0", label="η")
        ax2.set_ylabel("η", color="C0")
        ax.axhline(0, color="C3", ls="--")
        ax.set_xlabel("defect-only scale s_d")
        ax.set_ylabel("detuning (GHz)", color="C4")
    ax.set_title("(c) Defect-only fine trim")

    # (d) output-coupling Pareto: beta vs C, colored by eta
    ax = axes[1, 1]
    if output_rows:
        bv = _col(output_rows, "beta_wg")
        cv = _col(output_rows, "C_atom")
        ev = _col(output_rows, "eta_atom")
        sc = ax.scatter(bv, cv, c=ev, cmap="viridis", s=80, zorder=3)
        for r in output_rows:
            ax.annotate(f"{r['value']:.2f}", (r["beta_wg"], r["C_atom"]),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.axhline(20, color="k", ls=":", alpha=0.6, label="C=20 floor")
        ax.axvline(0.80, color="k", ls="--", alpha=0.6, label="β=0.80 floor")
        fig.colorbar(sc, ax=ax, label="η")
        ax.set_xlabel("β = κ_wg/κ_tot")
        ax.set_ylabel("cooperativity C")
        ax.legend(fontsize=8)
    ax.set_title("(d) Output-coupling tradeoff (label = hole-size scale)")
    fig.suptitle("Fig. 5 — Tuning sweeps and the overcoupling Pareto map",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, out_path)


# ── Fig 6: atom-position robustness heatmaps ───────────────────────────────

def fig_position_heatmaps(yz, xz, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    def _hm(ax, npz, xkey, xlabel, field, label, mark_x):
        Z = np.asarray(npz["z_heights_nm"])
        X = np.asarray(npz[xkey])
        A = np.asarray(npz[field])               # (nz, nx)
        im = ax.pcolormesh(X, Z, A, cmap="viridis", shading="auto")
        ax.scatter([mark_x], [250], marker="*", s=140, color="red",
                   edgecolor="white", zorder=5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("atom height above surface (nm)")
        ax.set_title(label)
        fig.colorbar(im, ax=ax, fraction=0.046)

    _hm(axes[0, 0], yz, "y_offsets_nm", "lateral offset y (nm)", "eta",
        "η vs height & lateral offset", 0.0)
    _hm(axes[0, 1], yz, "y_offsets_nm", "lateral offset y (nm)", "C",
        "C vs height & lateral offset", 0.0)
    _hm(axes[1, 0], xz, "x_offsets_nm", "along-beam offset x (nm)", "eta",
        "η vs height & along-beam offset", 0.0)
    _hm(axes[1, 1], xz, "x_offsets_nm", "along-beam offset x (nm)", "C",
        "C vs height & along-beam offset", 0.0)
    fig.suptitle("Fig. 6 — Atom-position robustness (★ = nominal 250 nm)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, out_path)


# ── Fig 7: fabrication robustness ──────────────────────────────────────────

def fig_fab_robustness(tol_rows, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    det_un = _col(tol_rows, "detuning_unretuned_GHz")
    det_re = _col(tol_rows, "detuning_retuned_GHz")
    eta = _col(tol_rows, "eta_atom")
    beta = _col(tol_rows, "beta_wg")

    axes[0, 0].hist(det_un, bins=12, color="C0", alpha=0.8)
    axes[0, 0].axvline(0, color="C3", ls="--")
    axes[0, 0].set_xlabel("detuning from D2 (GHz)")
    axes[0, 0].set_title("Unretuned detuning under fab perturbations")

    axes[0, 1].hist(det_re, bins=12, color="C2", alpha=0.8)
    axes[0, 1].axvline(0, color="C3", ls="--")
    axes[0, 1].set_xlabel("detuning after global-scale retune (GHz)")
    axes[0, 1].set_title("Retuned detuning (trim recovers D2)")

    axes[1, 0].hist(eta, bins=12, color="C1", alpha=0.8)
    axes[1, 0].axvline(0.75, color="k", ls=":", label="η=0.75 floor")
    axes[1, 0].set_xlabel("η")
    axes[1, 0].set_title("η robustness")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].hist(beta, bins=12, color="C4", alpha=0.8)
    axes[1, 1].axvline(0.80, color="k", ls=":", label="β=0.80 floor")
    axes[1, 1].set_xlabel("β")
    axes[1, 1].set_title("β robustness")
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("Fig. 7 — Fabrication-tolerance robustness (corner grid)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, out_path)


# ── Fig 8: five-design comparison ──────────────────────────────────────────

def fig_comparison_bars(rows, out_path):
    by = {r["design"]: r for r in rows}
    designs = [d for d in DESIGN_ORDER if d in by]
    colors = [DESIGN_COLORS[d] for d in designs]
    metrics = [("eta_atom", "η", False), ("beta_wg", "β", False),
               ("C_atom", "C (log)", True), ("Q", "Q (log)", True),
               ("Vmode_lambda_n3", "V (λ/n)³", False),
               ("detuning_GHz", "|detuning| (GHz, log)", True)]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (key, label, logy) in zip(axes.ravel(), metrics):
        vals = [abs(by[d][key]) if key == "detuning_GHz" else by[d][key]
                for d in designs]
        ax.bar(range(len(designs)), vals, color=colors)
        ax.set_xticks(range(len(designs)))
        ax.set_xticklabels(designs, rotation=30, ha="right", fontsize=8)
        ax.set_title(label)
        if logy:
            ax.set_yscale("log")
        if key == "eta_atom":
            ax.axhline(0.75, color="k", ls=":", alpha=0.6)
        if key == "beta_wg":
            ax.axhline(0.80, color="k", ls=":", alpha=0.6)
    fig.suptitle("Fig. 8 — Five-design comparison (one metric convention)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, out_path)


# ── Supplements ────────────────────────────────────────────────────────────

def fig_convergence(mesh_rows, pml_rows, rt_rows, out_path):
    axes_data = [
        ("mesh_dl_um", mesh_rows, "defect mesh dl (µm)"),
        ("pml_layers", pml_rows, "PML layers"),
        ("run_time_s", rt_rows, "run time (s)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (_, rows, xlabel) in zip(axes, axes_data):
        if not rows:
            ax.set_visible(False)
            continue
        v = _col(rows, "value")
        ax.plot(v, _col(rows, "detuning_GHz"), "o-", color="C0", label="detuning (GHz)")
        ax2 = ax.twinx()
        ax2.plot(v, _col(rows, "eta_atom"), "s--", color="C3", label="η")
        ax2.plot(v, _col(rows, "beta_wg"), "^:", color="C1", label="β")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("detuning from D2 (GHz)", color="C0")
        ax2.set_ylabel("η, β")
        ax.grid(alpha=0.25)
    axes[0].set_title("Mesh convergence")
    axes[1].set_title("PML convergence")
    axes[2].set_title("Run-time / fit-window convergence")
    fig.suptitle("Supplement — Numerical convergence of the final design", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, out_path)


def fig_mode_isolation(spec_freq_hz, spec_amp, modescan_rows, f_sel_thz,
                       kappa_2pi_hz, out_path):
    fig, ax = plt.subplots(figsize=(11, 4.6))
    if len(spec_freq_hz):
        ax.plot(np.asarray(spec_freq_hz) / 1e12,
                np.asarray(spec_amp) / np.max(spec_amp), color="C0", lw=1,
                label="combined-signal spectrum")
    ax.axvline(F_D2_THZ, color="C3", ls="--", label="Rb D2 (384.23)")
    ax.axvline(f_sel_thz, color="C2", ls=":", label=f"cavity mode ({f_sel_thz:.4f})")
    for r in modescan_rows:
        if not r["is_selected_mode"]:
            ax.axvline(r["freq_THz"], color="gray", ls="-", alpha=0.4)
    lw_ghz = kappa_2pi_hz / 1e9
    ax.set_xlabel("frequency (THz)")
    ax.set_ylabel("normalized spectral amplitude")
    ax.set_title(f"Mode isolation — single cavity mode; loaded linewidth "
                 f"κ/2π = {lw_ghz:.1f} GHz")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, out_path)
