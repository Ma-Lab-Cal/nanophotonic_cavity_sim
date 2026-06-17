"""Machine-readable dataset writers with fixed schemas.

Single source of truth for the on-disk data formats so every figure depends
only on a saved file, never on live simulation state. csv + json + npz; a
``convention`` block is embedded in the headline json files for traceability.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from inverse_design import config as cfg

SCHEMA_VERSION = "1.0"


def convention_block() -> dict:
    """The single metric convention used across every dataset and figure."""
    return {
        "schema_version": SCHEMA_VERSION,
        "metrics_path": "inverse_design.diagnostics.perf_from_simulation",
        "f_target_Hz": cfg.F_TARGET,
        "f_target_THz": cfg.F_TARGET / 1e12,
        "gamma_2pi_MHz": cfg.GAMMA_RB / (2 * np.pi * 1e6),
        "fiber_efficiency": cfg.FIBER_EFFICIENCY,
        "dipole_moment_Cm": cfg.DIPOLE_MOMENT,
        "g_definition": "g = d*sqrt(omega*u_pol/(2*hbar*eps0*Integral[eps|E|^2]dV)), reported as g/2pi in Hz",
        "g_transition_note": ("d = <J=1/2||er||J'=3/2> = 3.58e-29 C*m (J reduced dipole, Steck). "
                              "For a specific |F,mF>->|F',mF'> transition and polarization, "
                              "g = g_reduced * dipole_factor * |e.E_hat|, dipole_factor<=1 the "
                              "cycling-normalized amplitude (1.000 sigma+ cycling; 0.775=sqrt(3/5); "
                              "0.394=sqrt(7/45); 0.577=1/sqrt(3) line-averaged). g is NOT universal."),
        "C_definition": "C = 4 g^2 / (kappa_tot * gamma)",
        "eta_definition": "eta = beta * C/(C+1) * eta_fiber",
        "eta_chip_definition": "eta_chip = beta * C/(C+1)  (no fiber factor)",
        "eta_with_fiber_definition": "eta_with_fiber = eta_chip * eta_fiber, eta_fiber=0.99 ASSUMED external collection",
        "beta_definition": "HEADLINE beta = beta_flux = kappa(-x)/kappa_directional(total) (directional flux ratio)",
        "beta_guided_definition": ("beta_guided = beta_flux * (power in all guided modes / total plane power), "
                                   "from a mode-expansion ModeMonitor co-located with a flux monitor in the "
                                   "output (-x) waveguide; beta_fund uses the fundamental mode only. "
                                   "Ordering beta_flux >= beta_guided >= beta_fund."),
        "Vmode_definition": "V = Integral[eps|E|^2]dV / max(eps|E|^2), in (lambda/n)^3, NO legacy symmetry factor",
        "Vmode_dispersive_note": ("legacy uses eps|E|^2; the dispersive convention replaces eps by "
                                  "Re[d(omega*eps)/domega] in SiN (a ~1-2% correction, reported separately)"),
        "kappa_tot": "linewidth kappa = 2*pi*f_res/Q (ResonanceFinder ringdown); NOTE != kappa_directional(total)",
        "f_res_numerical_note": ("absolute f_res carries a mesh numerical uncertainty; the converged value "
                                 "and +/- delta come from the full-beam-refined mesh ladder (validate_design.py, "
                                 "freq_resolution.json), not a single production mesh"),
        "output_face": cfg.OUTPUT_FACE,
        "atom_offsets_nm": [int(z * 1000) for z in cfg.ATOM_SURFACE_OFFSETS_UM],
        "thickness_um": cfg.BASELINE_CONTEXT["thickness"],
        "width_um": cfg.BASELINE_CONTEXT["width"],
        "scope": ("EM-derived scalar proxies (fixed-position, two-level, dipole-aligned); "
                  "not a cavity-QED / master-equation / quantum-trajectory simulation"),
    }


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path, obj, with_convention=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _jsonable(obj)
    if with_convention and isinstance(payload, dict):
        payload = {"convention": convention_block(), **payload}
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
    return path


def write_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({c: _scalar(r.get(c)) for c in columns})
    return path


def _scalar(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    return v


def read_csv(path) -> list:
    """Read a csv back to a list of dicts, coercing numeric/bool fields."""
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            d = {}
            for k, v in row.items():
                d[k] = _coerce(v)
            out.append(d)
    return out


def read_json(path):
    with open(path) as f:
        return json.load(f)


def _coerce(v):
    if v is None or v == "":
        return v
    if v in ("True", "False"):
        return v == "True"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in arrays.items()})
    return path


# ── derived helpers ────────────────────────────────────────────────────────

def perf_row(label: str, perf: dict, source: str = "") -> dict:
    """Flatten a perf dict into a comparison-table row."""
    detuning_ghz = (perf["f_res_Hz"] - cfg.F_TARGET) / 1e9
    return {
        "design": label,
        "f_res_THz": perf["f_res_Hz"] / 1e12,
        "detuning_GHz": detuning_ghz,
        "eta_atom": perf["eta_atom"],
        "beta_wg": perf["beta_wg"],
        "Q": perf["Q"],
        "kappa_tot_2pi_GHz": perf["kappa_tot_2pi_GHz"],
        "C_atom": perf["C_atom"],
        "Vmode_lambda_n3": perf["Vmode_lambda_n3"],
        "g_2pi_GHz_at_250nm": perf["g_2pi_Hz_at_250nm"] / 1e9,
        "source": source,
    }


COMPARISON_COLUMNS = [
    "design", "f_res_THz", "detuning_GHz", "eta_atom", "beta_wg", "Q",
    "kappa_tot_2pi_GHz", "C_atom", "Vmode_lambda_n3", "g_2pi_GHz_at_250nm",
    "source",
]

TUNING_COLUMNS = [
    "knob", "value", "f_res_THz", "detuning_GHz", "Q", "kappa_tot_2pi_GHz",
    "beta_wg", "C_atom", "eta_atom", "Vmode_lambda_n3", "g_2pi_GHz_at_250nm",
    "design_marker", "is_selected", "hdf5",
]

TOLERANCE_COLUMNS = [
    "sample_id", "mode", "etch_nm", "lattice_nm", "width_nm",
    "f_res_THz", "detuning_unretuned_GHz", "detuning_retuned_GHz",
    "retune_scale", "Q", "beta_wg", "C_atom", "eta_atom", "hdf5",
]

CONVERGENCE_COLUMNS = [
    "axis", "value", "f_res_THz", "detuning_GHz", "Q", "kappa_tot_2pi_GHz",
    "beta_wg", "C_atom", "eta_atom", "Vmode_lambda_n3",
]

MODE_SCAN_COLUMNS = [
    "freq_THz", "detuning_GHz", "Q", "amplitude", "error", "is_selected_mode",
    "separation_linewidths",
]
