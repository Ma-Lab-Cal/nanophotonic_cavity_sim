"""Atom-position g/C/η maps and the loss-budget decomposition.

Both are pure postprocessing of a single completed verification sim — no extra
cloud cost. They reuse the exact ``inverse_design.diagnostics`` convention:
g = d·√(ω·u_pol/(2ħε₀∫u dV)) sampled on the x=0 yz profile monitor, C =
4g²/(κγ), η = β·C/(C+1)·η_fiber. Call them only AFTER
``diagnostics.perf_from_simulation`` has populated the wide-window resonance/κ
caches on the same sim object.
"""

from __future__ import annotations

import numpy as np
from scipy.constants import epsilon_0, hbar

from inverse_design import config as cfg


def _total_energy_integral_m3(sim_obj) -> float:
    u_tot = sim_obj.energy_density
    return float(np.abs(u_tot).integrate(coord=("x", "y", "z"))) * 1e-18


def g_C_eta_grid(sim_obj, beta_wg: float, z_heights_nm, y_offsets_nm,
                 dipole_moment=cfg.DIPOLE_MOMENT, pol="Ey",
                 fiber_efficiency=cfg.FIBER_EFFICIENCY) -> dict:
    """g/2π (Hz), C, η on a (z_height × y_offset) grid above the beam surface.

    z heights are offsets above the top surface (z = thickness/2 + dz); y offsets
    are lateral positions in the beam (0 = center). κ and β are held from the
    cavity (the plan permits this — η,C are re-derived only from the local
    field), so this is a field postprocess, not a per-position re-solve.
    """
    omega = sim_obj.resonant_omega_c
    kappa_tot = float(sim_obj.kappa_tot)
    U_int_m3 = _total_energy_integral_m3(sim_obj)
    thickness = sim_obj.context["thickness"]

    # symmetry-expanded access (sim_data[...] applies the y-symmetry so the
    # map is populated for y<0; .monitor_data[...] would only hold y>=0)
    prof = sim_obj.sim_data["Field_Profile_Monitor_x"]
    E_pol = np.abs(getattr(prof, pol)) ** 2
    dt = float(E_pol.t[-1] - E_pol.t[0])
    u_prof = (E_pol.integrate(coord="t") / dt).squeeze(drop=True)   # dims (y, z)

    z_abs = thickness / 2 + np.asarray(z_heights_nm, dtype=float) * 1e-3
    y_abs = np.asarray(y_offsets_nm, dtype=float) * 1e-3

    u_grid = u_prof.interp(y=y_abs, z=z_abs)            # xarray dims (y, z)
    u_vals = np.clip(np.asarray(u_grid.transpose("z", "y")), 0.0, None)  # (nz, ny)

    g = dipole_moment * np.sqrt(omega * u_vals / (2 * hbar * epsilon_0 * U_int_m3))
    g_2pi = g / (2 * np.pi)
    C = 4 * g ** 2 / (kappa_tot * cfg.GAMMA_RB)
    eta = beta_wg * C / (C + 1) * fiber_efficiency

    return {
        "z_heights_nm": np.asarray(z_heights_nm, dtype=float),
        "y_offsets_nm": np.asarray(y_offsets_nm, dtype=float),
        "g_2pi_Hz": g_2pi,                # (nz, ny)
        "C": C,                           # (nz, ny)
        "eta": eta,                       # (nz, ny)
        "beta_wg_held": float(beta_wg),
        "kappa_tot_2pi_GHz_held": kappa_tot / (2 * np.pi * 1e9),
        "nominal_z_nm": 250.0,
        "nominal_y_nm": 0.0,
    }


def g_C_eta_grid_axial(sim_obj, beta_wg: float, z_heights_nm, x_offsets_nm,
                       dipole_moment=cfg.DIPOLE_MOMENT, pol="Ey",
                       fiber_efficiency=cfg.FIBER_EFFICIENCY) -> dict:
    """g/2π, C, η on a (z_height × x_along_beam) grid at y=0.

    Uses the xz profile monitor (Field_Profile_Monitor_y) so this maps the
    along-beam robustness of the coupling away from the antinode (x=0). Same
    convention and denominator as ``g_C_eta_grid``.
    """
    omega = sim_obj.resonant_omega_c
    kappa_tot = float(sim_obj.kappa_tot)
    U_int_m3 = _total_energy_integral_m3(sim_obj)
    thickness = sim_obj.context["thickness"]

    prof = sim_obj.sim_data["Field_Profile_Monitor_y"]
    E_pol = np.abs(getattr(prof, pol)) ** 2
    dt = float(E_pol.t[-1] - E_pol.t[0])
    u_prof = (E_pol.integrate(coord="t") / dt).squeeze(drop=True)   # dims (x, z)

    z_abs = thickness / 2 + np.asarray(z_heights_nm, dtype=float) * 1e-3
    x_abs = np.asarray(x_offsets_nm, dtype=float) * 1e-3
    u_grid = u_prof.interp(x=x_abs, z=z_abs)            # dims (x, z)
    u_vals = np.clip(np.asarray(u_grid.transpose("z", "x")), 0.0, None)

    g = dipole_moment * np.sqrt(omega * u_vals / (2 * hbar * epsilon_0 * U_int_m3))
    g_2pi = g / (2 * np.pi)
    C = 4 * g ** 2 / (kappa_tot * cfg.GAMMA_RB)
    eta = beta_wg * C / (C + 1) * fiber_efficiency
    return {
        "z_heights_nm": np.asarray(z_heights_nm, dtype=float),
        "x_offsets_nm": np.asarray(x_offsets_nm, dtype=float),
        "g_2pi_Hz": g_2pi, "C": C, "eta": eta,
        "beta_wg_held": float(beta_wg),
        "kappa_tot_2pi_GHz_held": kappa_tot / (2 * np.pi * 1e9),
        "nominal_z_nm": 250.0, "nominal_x_nm": 0.0,
    }


def loss_budget(sim_obj) -> dict:
    """Decompose the cavity loss into the waveguide channel vs residual.

    Two complementary estimators are reported (they need not agree exactly):
      * linewidth κ_tot from the resonance ring-down fit (sim_obj.kappa_tot),
      * the directional flux budget κ_dir{face}, whose parallel sum is the
        directional total. β = κ(−x)/κ_dir(total) (the perf convention).
    """
    kappa_dir = sim_obj.kappa_dir                       # rad/s per face
    out_face = cfg.OUTPUT_FACE
    twopi_ghz = 2 * np.pi * 1e9

    faces = {k: float(v) / twopi_ghz for k, v in kappa_dir.items()}
    k_wg = float(kappa_dir[out_face])
    k_tot_dir = float(kappa_dir["total"])
    k_res = k_tot_dir - k_wg

    # resonance fit residual of the selected mode
    df = sim_obj.resonance_df
    best = df.sort_values("Q").iloc[-1]
    fit_error = float(best.get("error", np.nan))
    fit_amp = float(best.get("amplitude", np.nan))

    return {
        "kappa_linewidth_2pi_GHz": float(sim_obj.kappa_tot) / twopi_ghz,
        "kappa_dir_total_2pi_GHz": k_tot_dir / twopi_ghz,
        "kappa_wg_2pi_GHz": k_wg / twopi_ghz,
        "kappa_residual_2pi_GHz": k_res / twopi_ghz,
        "beta_wg": k_wg / k_tot_dir,
        "residual_fraction": k_res / k_tot_dir,
        "kappa_per_face_2pi_GHz": faces,
        "output_face": out_face,
        "fit_error": fit_error,
        "fit_amplitude": fit_amp,
        "f_res_Hz": float(sim_obj.resonant_frequency),
        "Q_linewidth": float(sim_obj.Q),
    }


def resonance_spectrum(sim_obj) -> dict:
    """Fitted resonance table (freqs, Q, amplitude, error) for the spectrum plot."""
    df = sim_obj.resonance_df
    freqs = np.abs(np.asarray(df.index, dtype=float))
    return {
        "freq_Hz": freqs,
        "Q": np.asarray(df["Q"], dtype=float),
        "amplitude": np.asarray(df["amplitude"], dtype=float),
        "error": np.asarray(df["error"], dtype=float),
        "f_res_Hz": float(sim_obj.resonant_frequency),
        "Q_res": float(sim_obj.Q),
    }
