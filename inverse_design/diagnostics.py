"""Non-gradient verification of inverse-designed geometries.

Reuses the existing time-domain analysis pipeline in
``main_code.simulation.Cavity_simulation`` (ResonanceFinder Q, directional Q,
mode volume, decay rates) on geometry injected from the differentiable
parameterization, and adds an independent, units-checked computation of the
atom-photon coupling g(r), cooperativity C(r) and collection efficiency
eta(r) at the realistic atom positions above the beam.

This module is also used to evaluate the BASELINE design (design_1 from
cavities_780.ipynb) so that baseline and optimized designs are compared with
identical machinery.
"""

from __future__ import annotations

import numpy as np
import tidy3d as td
from scipy.constants import epsilon_0, hbar

from inverse_design import config as cfg
from inverse_design.parametrization import CavityParametrization

C0 = td.constants.C_0          # um/s
C0_M = C0 * 1e-6               # m/s


class FrozenGeometryCavitySimulation:
    """Lazy import wrapper so importing diagnostics never requires main_code."""
    def __new__(cls, *args, **kwargs):
        return _make_frozen_class()(*args, **kwargs)


def _make_frozen_class():
    from main_code.simulation import Cavity_simulation

    class _FrozenGeometryCavitySimulation(Cavity_simulation):
        """Cavity_simulation with injected (pre-built) hole structures.

        Only ``_build_nanobeam`` and ``_is_symmetric`` are overridden; all
        time-domain monitors and analysis methods run verbatim.
        """

        def __init__(self, structures, parameters, n_cells, context,
                     beam_layout, **kwargs):
            self._frozen_structures = structures
            super().__init__(parameters=parameters, n_cells=n_cells,
                             context=context, beam_layout=beam_layout, **kwargs)

        def _build_nanobeam(self):
            return self._frozen_structures

        def _is_symmetric(self):
            return False   # inverse-designed cavities are one-sided

        def _create_simulation(self, grid_size_override=(0.01, 0.01, 0.01)):
            # The legacy Field_Time_Monitor starts at run_time - 1/freq0; if
            # the solver shuts off early (default shutoff=1e-5) that window is
            # never reached and the energy-density analysis gets zero samples.
            # Disable early shutoff so the late-time window always exists.
            sim = super()._create_simulation(grid_size_override)
            return sim.updated_copy(shutoff=0.0)

    return _FrozenGeometryCavitySimulation


def verification_sim_from_layout(layout_np: dict, par: CavityParametrization,
                                 run_time: float = 5e-12,
                                 grid_size_override=(0.01, 0.01, 0.01)):
    """Build a FrozenGeometryCavitySimulation from a detached layout."""
    from inverse_design.builder import build_structures, sim_domain

    positions = np.asarray(layout_np["positions"])
    a = np.asarray(layout_np["a"])
    context = dict(par.context)

    _, size, _ = sim_domain(positions, a, context)
    structures = build_structures(layout_np, par, size[0])

    # Synthetic legacy-style dicts so base-class bookkeeping works.
    gp = np.stack([np.asarray(layout_np["sx"]), np.asarray(layout_np["sy"])], axis=1)
    beam_layout = {"positions": positions, "lattice": a, "geometry_params": gp}
    n_d = par.n_cells["N_defect"]
    i0 = par.n_cells["N_left_taper"] + par.n_cells["N_left_mirror"]
    defect_lat = float(np.mean(a[i0:i0 + n_d]))
    parameters = {
        key: {"lattice": defect_lat, "geometry_params": np.array([0.2, 0.2])}
        for key in ("parameters_taper_left", "parameters_mirrors_left",
                    "parameters_defect", "parameters_mirrors_right",
                    "parameters_taper_right")
    }

    sim = FrozenGeometryCavitySimulation(
        structures=structures, parameters=parameters, n_cells=dict(par.n_cells),
        context=context, beam_layout=beam_layout, run_time=run_time,
    )
    sim.build(grid_size_override=grid_size_override, num_modes=1, plot=False)
    return sim


def baseline_verification_sim(run_time: float = 5e-12,
                              grid_size_override=(0.01, 0.01, 0.01)):
    """Verification sim of the legacy baseline (design_1), via main_code.Cavity."""
    from main_code.cavity import Cavity

    cav = Cavity(n_cells=dict(cfg.BASELINE_N_CELLS),
                 parameters=cfg.BASELINE_PARAMETERS,
                 context=dict(cfg.BASELINE_CONTEXT))
    cav.build_simulation(grid_size_override=grid_size_override, num_modes=1,
                         plot=False)
    sim_obj = cav.simulation
    # same early-shutoff guard as the frozen verification sims
    sim_obj.sim = sim_obj.sim.updated_copy(shutoff=0.0)
    return sim_obj


# ──────────────────────────────────────────────────────────────────────────
# Independent g / C / eta computation (units-checked)
# ──────────────────────────────────────────────────────────────────────────

def g_at_points(sim_obj, dipole_moment=cfg.DIPOLE_MOMENT,
                pol: str = "Ey") -> dict:
    """g/2pi (Hz) at the atom positions above the beam surface.

    g(r) = d * sqrt( omega * u_pol(r) / (2 hbar eps0 * Integral[u] dV) ),
    with u_pol = |E_pol|^2 time-averaged (vacuum at the atom: eps_r = 1) and
    Integral[u] the time-averaged total field energy integral eps_r |E|^2.
    The ratio is normalization- and averaging-convention independent because
    both monitors cover the same late-time window.

    Numerator: the full-resolution x=src yz-plane profile monitor
    (Field_Profile_Monitor_x, ~10-30 nm z spacing near the atom, interpolated
    in z). The 3D Field_Time_Monitor is too coarse there (~150 nm z spacing)
    but fine for the volume integral (denominator).

    NOTE: tidy3d expands monitor data across symmetry planes on access
    (SimulationData returns symmetry_expanded_copy), so the full-domain
    integral needs NO extra symmetry factor. (The legacy mode_volume()
    multiplies one in — kept there for convention continuity; it cancels in
    baseline-vs-optimized comparisons but not in g, hence this independent
    computation.)
    """
    omega = sim_obj.resonant_omega_c           # rad/s

    # numerator: time-averaged |E_pol|^2 on the yz profile plane (vacuum)
    prof = sim_obj.sim_data.monitor_data["Field_Profile_Monitor_x"]
    E_pol = np.abs(getattr(prof, pol)) ** 2
    delta_t_p = E_pol.t[-1] - E_pol.t[0]
    u_prof = E_pol.integrate(coord="t") / delta_t_p     # dims (x,y,z)

    # denominator: total energy integral from the 3D monitor (same window)
    u_tot = sim_obj.energy_density             # eps |E|^2, time-averaged
    U_int_um3 = float(np.abs(u_tot).integrate(coord=("x", "y", "z")))
    U_int_m3 = U_int_um3 * 1e-18               # um^3 -> m^3

    thickness = sim_obj.context["thickness"]

    out = {}
    for k, dz in enumerate(cfg.ATOM_SURFACE_OFFSETS_UM):
        z_atom = thickness / 2 + dz
        u_here = float(u_prof.squeeze(drop=True).interp(y=0.0, z=z_atom))
        u_here = max(u_here, 0.0)
        g = dipole_moment * np.sqrt(omega * u_here / (2 * hbar * epsilon_0 * U_int_m3))
        out[f"g_2pi_Hz_at_{int(dz * 1000)}nm"] = g / (2 * np.pi)
    return out


def perf_from_simulation(sim_obj, fiber_efficiency=cfg.FIBER_EFFICIENCY,
                         freq_halfwindow=8e12) -> dict:
    """Full physical performance dict from a completed verification sim.

    The resonance search window is widened to +/- freq_halfwindow around the
    drive frequency: the default legacy window (freq0 +/- fwidth/2 ~ +/-2 THz)
    can miss the true mode when the resonance drifts during optimization.
    """
    f0 = sim_obj.context["freq0"]
    sim_obj.analyse_resonances(freq_window=(f0 - freq_halfwindow,
                                            f0 + freq_halfwindow))
    perf = {}
    perf["f_res_Hz"] = float(sim_obj.resonant_frequency)
    perf["Q"] = float(sim_obj.Q)
    perf["kappa_tot_2pi_GHz"] = float(sim_obj.kappa_tot / (2 * np.pi * 1e9))
    perf["Vmode_lambda_n3"] = float(sim_obj.Vmode)
    perf["purcell"] = float((3 / (4 * np.pi ** 2)) * (sim_obj.Q / sim_obj.Vmode))

    Q_dir = sim_obj.Q_directional
    perf["Q_directional"] = {k: float(np.asarray(v)) for k, v in Q_dir.items()}
    kappa = sim_obj.kappa_dir
    k_out = float(np.asarray(kappa.get(cfg.OUTPUT_FACE)))
    k_tot_dir = float(np.asarray(kappa.get("total")))
    perf["beta_wg"] = k_out / k_tot_dir

    gs = g_at_points(sim_obj)
    perf.update(gs)

    # C and eta at the nominal atom position (middle offset)
    mid = int(cfg.ATOM_SURFACE_OFFSETS_UM[1] * 1000)
    g_mid = gs[f"g_2pi_Hz_at_{mid}nm"] * 2 * np.pi          # rad/s
    kappa_tot = float(sim_obj.kappa_tot)                     # rad/s
    C = 4 * g_mid ** 2 / (kappa_tot * cfg.GAMMA_RB)
    perf["C_atom"] = float(C)
    perf["eta_atom"] = float(perf["beta_wg"] * C / (C + 1) * fiber_efficiency)
    perf["gamma_2pi_MHz"] = cfg.GAMMA_RB / (2 * np.pi * 1e6)
    perf["fiber_efficiency"] = fiber_efficiency
    return perf
