"""Differentiable objective: physics figure of merit + geometric penalties.

The optimization loss is a frequency-domain surrogate of the physical
photon-interface efficiency

    eta = (kappa_wg / kappa_tot) * C/(C+1) * eta_fiber

evaluated from a single dipole-driven forward simulation:

* ``P_wg(f)``  — power into the chosen output waveguide mode (ModeMonitor),
* ``P_tot(f)`` — total radiated power (closed box of FieldMonitors, flux),
* ``u_atom(f)``— |E_pol|^2 at realistic atom positions above the beam.

A softmax over the frequency window picks the resonance smoothly. The
branching ratio beta = P_wg/P_tot approximates kappa_wg/kappa_tot at
resonance, and u_atom/P_tot is monotonically related to the cooperativity
C = 4 g^2/(kappa*gamma) (both ~ Q/V_eff at the atom site); a single scalar
calibration c0 (from the baseline's time-domain analysis) converts it to an
absolute C estimate. Exact Q, kappa budget, V_mode, g, C, eta remain
diagnostics computed outside the gradient path (see diagnostics.py).
"""

from __future__ import annotations

import autograd.numpy as anp
import numpy as np

from inverse_design import config as cfg
from inverse_design.parametrization import CavityParametrization, softplus


# ──────────────────────────────────────────────────────────────────────────
# Geometric penalties (differentiable, no simulation required)
# ──────────────────────────────────────────────────────────────────────────

def _barrier(violation, scale=cfg.PENALTY_SCALE_UM):
    """Smooth one-sided quadratic barrier: ~0 when violation<0 (feasible)."""
    return softplus(violation / scale) ** 2


def penalties(par: CavityParametrization, layout, curvature: bool = False) -> dict:
    """Return dict of named penalty scalars (autograd-traceable)."""
    pos = layout["positions"]
    ext = par.extents(layout)
    width = par.context["width"]

    # hole-hole dielectric gap along x
    gaps = (pos[1:] - pos[:-1]) - (ext["x_half"][1:] + ext["x_half"][:-1])
    p_gap = anp.sum(_barrier(cfg.GAP_MIN_UM - gaps))

    # side rails: beam edge to hole edge (keeps the suspended beam connected)
    rails = width / 2.0 - ext["y_half"]
    p_rail = anp.sum(_barrier(cfg.RAIL_MIN_UM - rails))

    # lattice smoothness (gentle-confinement prior)
    a = layout["a"]
    a0 = anp.mean(a)
    p_smooth = anp.sum((a[2:] - 2 * a[1:-1] + a[:-2]) ** 2) / a0 ** 2

    out = {
        "gap": cfg.LAMBDA_GAP * p_gap,
        "rail": cfg.LAMBDA_RAIL * p_rail,
        "smooth": cfg.LAMBDA_SMOOTH * p_smooth,
    }

    if curvature:
        out["curv"] = cfg.LAMBDA_CURV * curvature_penalty(layout)
    return out


def curvature_penalty(layout, r_min: float = 0.040) -> float:
    """Penalize spline boundary radius of curvature below r_min (um).

    Discrete curvature from finite differences along each closed polygon.
    Each hole's contribution is gated by a smooth "does this hole exist"
    weight: holes shrunk below HOLE_VANISH_UM count as absent (that is how
    the optimizer removes holes) and are exempt from fabricability limits.
    """
    d_eff = anp.sqrt(layout["sx"] * layout["sy"])           # effective diameter
    gate = 1.0 / (1.0 + anp.exp(-(d_eff - cfg.HOLE_EXISTS_UM) / 0.005))
    total = 0.0
    for i, v in enumerate(layout["vertices"]):
        d1 = (anp.roll(v, -1, axis=0) - anp.roll(v, 1, axis=0)) / 2.0
        d2 = anp.roll(v, -1, axis=0) - 2 * v + anp.roll(v, 1, axis=0)
        num = anp.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
        den = (d1[:, 0] ** 2 + d1[:, 1] ** 2) ** 1.5 + 1e-12
        kappa = num / den                       # 1/um
        total = total + gate[i] * anp.mean(_barrier(kappa - 1.0 / r_min, scale=5.0))
    return total / len(layout["vertices"])


# ──────────────────────────────────────────────────────────────────────────
# Hard validity check (plain numpy, used as pre-upload gate)
# ──────────────────────────────────────────────────────────────────────────

def validate_layout(par: CavityParametrization, layout) -> list:
    """Return a list of human-readable problems (empty = valid)."""
    lay = CavityParametrization.detach(layout)
    problems = []
    pos = np.asarray(lay["positions"])
    if not np.all(np.isfinite(pos)):
        problems.append("non-finite hole positions")
    width = par.context["width"]
    for i, v in enumerate(lay["vertices"]):
        v = np.asarray(v)
        if not np.all(np.isfinite(v)):
            problems.append(f"hole {i}: non-finite vertices")
            continue
        if np.max(np.abs(v[:, 1])) > width / 2:
            problems.append(f"hole {i}: extends beyond beam width")
        try:
            from shapely.geometry import Polygon
            if not Polygon(v).is_valid:
                problems.append(f"hole {i}: self-intersecting polygon")
        except ImportError:
            pass
    x_half = [np.max(np.abs(np.asarray(v)[:, 0])) for v in lay["vertices"]]
    gaps = np.diff(pos) - (np.array(x_half[1:]) + np.array(x_half[:-1]))
    if np.any(gaps <= 0):
        problems.append(f"{int(np.sum(gaps <= 0))} overlapping hole pairs")
    return problems


# ──────────────────────────────────────────────────────────────────────────
# Differentiable physics objective from simulation data
# ──────────────────────────────────────────────────────────────────────────

# Outward-normal signs for the six box faces (FieldData.flux is along +axis)
_FACE_SIGNS = {"box_+x": +1.0, "box_-x": -1.0, "box_+y": +1.0,
               "box_-y": -1.0, "box_+z": +1.0, "box_-z": -1.0}


def physics_metrics(sim_data, sim_cfg: cfg.SimConfig) -> dict:
    """Extract differentiable spectra from forward-simulation data."""
    eps = 1e-30

    # output waveguide mode power vs frequency
    amps = sim_data["wg_out"].amps.sel(direction=cfg.OUTPUT_DIRECTION,
                                       mode_index=0).values
    P_wg = anp.abs(amps) ** 2                                  # (Nf,)

    # total radiated power through the closed monitor box
    P_tot = 0.0
    for name, sign in _FACE_SIGNS.items():
        P_tot = P_tot + sign * sim_data[name].flux.values
    P_tot = anp.abs(P_tot) + eps                               # (Nf,)

    # |E_pol|^2 at the atom positions (weighted over heights)
    pol = "Ey"
    u_atom = 0.0
    for k, w in enumerate(cfg.ATOM_Z_WEIGHTS):
        E = sim_data[f"atom_{k}"].field_components[pol].values.ravel()
        u_atom = u_atom + w * anp.abs(E) ** 2                  # (Nf,)

    freqs = np.asarray(sim_cfg.freqs)

    # smooth peak picking over the window
    logp = anp.log(P_wg + eps)
    w_f = anp.exp(cfg.SOFTMAX_TAU * (logp - anp.max(logp)))
    w_f = w_f / anp.sum(w_f)

    f_hat = anp.sum(w_f * freqs)
    beta = anp.sum(w_f * P_wg / P_tot)
    C_raw = anp.sum(w_f * u_atom / P_tot)

    return {"P_wg": P_wg, "P_tot": P_tot, "u_atom": u_atom, "w_f": w_f,
            "f_hat": f_hat, "beta": beta, "C_raw": C_raw}


def loss_from_metrics(metrics: dict, par: CavityParametrization, layout,
                      c0: float, curvature: bool = False):
    """Scalar loss + aux dict from extracted metrics and geometry."""
    C_tilde = c0 * metrics["C_raw"]
    eta_tilde = metrics["beta"] * C_tilde / (C_tilde + 1.0)

    L_phys = -anp.log(eta_tilde + 1e-12)
    L_freq = cfg.W_FREQ * ((metrics["f_hat"] - cfg.F_TARGET) / cfg.FREQ_TOL_HZ) ** 2

    pens = penalties(par, layout, curvature=curvature)
    L_pen = sum(pens.values())

    loss = L_phys + L_freq + L_pen

    aux = {
        "eta_tilde": eta_tilde, "C_tilde": C_tilde, "beta": metrics["beta"],
        "C_raw": metrics["C_raw"], "f_hat": metrics["f_hat"],
        "L_phys": L_phys, "L_freq": L_freq, "penalties": pens,
    }
    return loss, aux
