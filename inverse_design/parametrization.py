"""Differentiable cavity parameterization.

Maps an unconstrained parameter vector ``theta`` (dict of autograd-traceable
arrays) to the physical cavity layout:

* ``a``   — per-cell lattice constants (hole spacings). Hole positions are the
  cumulative sum of these spacings, so the placement schedule is free.
* ``sx``, ``sy`` — per-hole shape scales. At circular spline control points
  (rho = 1) the hole is exactly an ellipse with diameters (sx, sy), matching
  the existing ``crystal_polygon_2d`` "ellipse" convention.
* ``rho`` — closed-spline shape control points: a uniform periodic cubic
  B-spline of the radius function r(phi). Radial (star-shaped) splines are
  closed and can never self-intersect. x- and y-mirror symmetry is enforced by
  tying control points, and the per-hole shape is a fixed smooth blend of
  N_ANCHORS anchor shapes (taper-L, mirror-L, defect, mirror-R, taper-R)
  following the same cubic blend profile as ``main_code.defect``.

All math that must be differentiated uses ``autograd.numpy``.
"""

from __future__ import annotations

import autograd.numpy as anp
import numpy as np
from autograd.scipy.special import logsumexp
from autograd.tracer import getval

from inverse_design import config as cfg


# ──────────────────────────────────────────────────────────────────────────
# Periodic cubic B-spline basis (fixed matrix, plain numpy)
# ──────────────────────────────────────────────────────────────────────────

def make_periodic_bspline_basis(n_ctrl: int = cfg.N_CTRL,
                                n_pts: int = cfg.N_SHAPE_PTS) -> np.ndarray:
    """Return a (n_pts, n_ctrl) matrix B with r(phi_j) = B @ rho.

    Uniform periodic cubic B-spline evaluated at n_pts equally spaced
    parameter values. Rows sum to 1 (partition of unity), so rho = const c
    gives r = c exactly.
    """
    B = np.zeros((n_pts, n_ctrl))
    ts = np.linspace(0.0, n_ctrl, n_pts, endpoint=False)
    for j, t in enumerate(ts):
        seg = int(np.floor(t))
        u = t - seg
        # uniform cubic B-spline blending functions
        b = np.array([
            (1 - u) ** 3 / 6.0,
            (3 * u ** 3 - 6 * u ** 2 + 4) / 6.0,
            (-3 * u ** 3 + 3 * u ** 2 + 3 * u + 1) / 6.0,
            u ** 3 / 6.0,
        ])
        for k in range(4):
            B[j, (seg - 1 + k) % n_ctrl] += b[k]
    return B


# Control point k sits at angle 2*pi*k/N_CTRL. Mirror symmetry in x
# (phi -> -phi) and y (phi -> pi - phi) ties the 12 control points into 4
# orbits: {0,6}, {1,5,7,11}, {2,4,8,10}, {3,9}.
_SYMMETRY_INDEX_MAP = np.array([0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1])
assert len(_SYMMETRY_INDEX_MAP) == cfg.N_CTRL
assert _SYMMETRY_INDEX_MAP.max() + 1 == cfg.N_FREE_RHO


def expand_symmetric_rho(rho_free):
    """(..., N_FREE_RHO) free control radii -> (..., N_CTRL) full ring."""
    return rho_free[..., _SYMMETRY_INDEX_MAP]


# ──────────────────────────────────────────────────────────────────────────
# Smooth unconstrained <-> physical transforms
# ──────────────────────────────────────────────────────────────────────────

def sigmoid_bounded(x, lo, hi):
    return lo + (hi - lo) / (1.0 + anp.exp(-x))


def inverse_sigmoid_bounded(v, lo, hi, eps=1e-6):
    v = np.clip((np.asarray(v, dtype=float) - lo) / (hi - lo), eps, 1 - eps)
    return np.log(v / (1 - v))


def softplus(x):
    # numerically stable softplus
    return anp.logaddexp(0.0, x)


# ──────────────────────────────────────────────────────────────────────────
# Anchor blend weights (fixed, plain numpy)
# ──────────────────────────────────────────────────────────────────────────

def make_anchor_weights(n_cells: dict) -> np.ndarray:
    """(N_holes, N_ANCHORS) fixed blend weights for the shape schedule.

    Anchors: 0=taper_left, 1=mirror_left, 2=defect_center, 3=mirror_right,
    4=taper_right. Tapers blend linearly mirror->taper outward (t=0 at the
    mirror-adjacent cell). The defect uses the legacy cubic blend
    2x^3 - 3x^2 + 1 from main_code.defect.defect_function (blend=1 at the
    defect center, 0 at the mirrors).
    """
    n_lt = n_cells["N_left_taper"]
    n_lm = n_cells["N_left_mirror"]
    n_d = n_cells["N_defect"]
    n_rm = n_cells["N_right_mirror"]
    n_rt = n_cells["N_right_taper"]
    n_half = n_d // 2

    rows = []

    # left taper: ordered left->right, outermost first. Legacy Taper uses
    # t = linspace(0,1,n) from mirror to taper tip; left section is flipped,
    # so the leftmost hole has t=1 (full taper) and the innermost t=0.
    ts = np.linspace(0, 1, n_lt) if n_lt > 1 else np.array([1.0])
    for t in ts[::-1]:
        w = np.zeros(cfg.N_ANCHORS)
        w[0], w[1] = t, 1 - t
        rows.append(w)

    for _ in range(n_lm):
        w = np.zeros(cfg.N_ANCHORS)
        w[1] = 1.0
        rows.append(w)

    # defect: left half blends mirror_left <-> defect, right half
    # mirror_right <-> defect, cubic in normalized index (legacy convention:
    # index 1 at the center-most cell, n_half at the mirror-adjacent cell).
    for side, n_side, anchor_mirror in ((0, n_half, 1), (1, n_d - n_half, 3)):
        idx = np.arange(1, n_side + 1)
        if side == 0:
            idx = idx[::-1]   # left half ordered outermost -> innermost
        for i in idx:
            x = i / n_half
            blend = np.clip(2 * x ** 3 - 3 * x ** 2 + 1, 0.0, 1.0)
            w = np.zeros(cfg.N_ANCHORS)
            w[2] = blend
            w[anchor_mirror] = 1 - blend
            rows.append(w)

    for _ in range(n_rm):
        w = np.zeros(cfg.N_ANCHORS)
        w[3] = 1.0
        rows.append(w)

    ts = np.linspace(0, 1, n_rt) if n_rt > 1 else np.array([1.0])
    for t in ts:
        w = np.zeros(cfg.N_ANCHORS)
        w[4], w[3] = t, 1 - t
        rows.append(w)

    W = np.vstack(rows)
    assert W.shape == (n_lt + n_lm + n_d + n_rm + n_rt, cfg.N_ANCHORS)
    assert np.allclose(W.sum(axis=1), 1.0)
    return W


# ──────────────────────────────────────────────────────────────────────────
# Main parameterization
# ──────────────────────────────────────────────────────────────────────────

class CavityParametrization:
    """theta (unconstrained dict) <-> physical hole layout."""

    def __init__(self, n_cells=None, context=None):
        self.n_cells = dict(n_cells or cfg.BASELINE_N_CELLS)
        self.context = dict(context or cfg.BASELINE_CONTEXT)
        self.n_holes = sum(self.n_cells.values())
        self.basis = make_periodic_bspline_basis()          # (n_pts, n_ctrl)
        self.anchor_weights = make_anchor_weights(self.n_cells)  # (N, n_anchors)
        self.angles = np.linspace(0, 2 * np.pi, cfg.N_SHAPE_PTS, endpoint=False)
        self._cos = np.cos(self.angles)
        self._sin = np.sin(self.angles)
        # index of the hole pair straddling the defect center (even N_defect)
        # or the central hole (odd N_defect)
        n_left = (self.n_cells["N_left_taper"] + self.n_cells["N_left_mirror"]
                  + self.n_cells["N_defect"] // 2)
        self._center_left_idx = n_left - 1 if self.n_cells["N_defect"] % 2 == 0 else n_left

    # ── theta construction ────────────────────────────────────────────

    def init_theta_from_baseline(self, parameters=None, n_cells=None) -> dict:
        """Unconstrained theta that exactly reproduces the legacy layout."""
        from main_code.cavity import Cavity

        parameters = parameters or cfg.BASELINE_PARAMETERS
        n_cells = n_cells or self.n_cells

        cav = Cavity(n_cells=dict(n_cells), parameters=parameters,
                     context=dict(self.context))
        layout = cav.beam_layout
        a = np.asarray(layout["lattice"], dtype=float)
        gp = np.atleast_2d(np.asarray(layout["geometry_params"], dtype=float))
        sx, sy = gp[:, 0], gp[:, 1]

        theta = {
            "a": inverse_sigmoid_bounded(a, *cfg.A_BOUNDS),
            "sx": inverse_sigmoid_bounded(sx, *cfg.SX_BOUNDS),
            "sy": inverse_sigmoid_bounded(sy, *cfg.SY_BOUNDS),
            # rho = 1 (circular ring -> exact ellipse) for all anchors
            "rho": inverse_sigmoid_bounded(
                np.ones((cfg.N_ANCHORS, cfg.N_FREE_RHO)), *cfg.RHO_BOUNDS),
        }
        self._baseline_positions = np.asarray(layout["positions"], dtype=float)
        return theta

    # ── differentiable mappings ───────────────────────────────────────

    def to_physical(self, theta) -> dict:
        a = sigmoid_bounded(theta["a"], *cfg.A_BOUNDS)
        sx = sigmoid_bounded(theta["sx"], *cfg.SX_BOUNDS)
        sy = sigmoid_bounded(theta["sy"], *cfg.SY_BOUNDS)
        rho_anchors = sigmoid_bounded(theta["rho"], *cfg.RHO_BOUNDS)
        # per-hole control radii: (N, n_free) = (N, n_anchors) @ (n_anchors, n_free)
        rho_holes = self.anchor_weights @ rho_anchors
        return {"a": a, "sx": sx, "sy": sy, "rho_anchors": rho_anchors,
                "rho_holes": rho_holes}

    def positions(self, a):
        """Hole centers from per-cell spacings; defect center pinned at x=0."""
        x = anp.cumsum(a) - a / 2.0
        ic = self._center_left_idx
        if self.n_cells["N_defect"] % 2 == 0:
            x_ref = (x[ic] + x[ic + 1]) / 2.0
        else:
            x_ref = x[ic]
        return x - x_ref

    def hole_radii(self, rho_holes):
        """(N, n_pts) radius samples r_ij = B @ rho_i (unit scale)."""
        rho_full = expand_symmetric_rho(rho_holes)           # (N, N_CTRL)
        return rho_full @ self.basis.T                       # (N, n_pts)

    def hole_vertices(self, phys) -> list:
        """List of (n_pts, 2) vertex arrays, hole-centered coordinates."""
        r = self.hole_radii(phys["rho_holes"])               # (N, n_pts)
        vx = 0.5 * phys["sx"][:, None] * r * self._cos[None, :]
        vy = 0.5 * phys["sy"][:, None] * r * self._sin[None, :]
        return [anp.stack([vx[i], vy[i]], axis=1) for i in range(self.n_holes)]

    def layout(self, theta) -> dict:
        """Full differentiable layout: positions + per-hole vertices."""
        phys = self.to_physical(theta)
        pos = self.positions(phys["a"])
        verts = self.hole_vertices(phys)
        return {"positions": pos, "a": phys["a"], "sx": phys["sx"],
                "sy": phys["sy"], "rho_holes": phys["rho_holes"],
                "vertices": verts}

    # ── geometry summary quantities used by penalties ─────────────────

    def extents(self, layout) -> dict:
        """Smooth per-hole x/y half-extents via logsumexp (soft max)."""
        # Soft max over vertices. logsumexp overestimates the true max by at
        # most log(n_pts)/beta ~ 4 nm at beta=1000 -- a small conservative
        # bias in the gap/rail constraints.
        beta = 1000.0  # 1/um
        xext, yext = [], []
        for v in layout["vertices"]:
            xext.append(logsumexp(beta * anp.abs(v[:, 0])) / beta)
            yext.append(logsumexp(beta * anp.abs(v[:, 1])) / beta)
        return {"x_half": anp.stack(xext), "y_half": anp.stack(yext)}

    # ── utilities ─────────────────────────────────────────────────────

    @staticmethod
    def detach(obj):
        """Strip autograd tracing -> plain numpy (for plots/GDS/serialization)."""
        if isinstance(obj, dict):
            return {k: CavityParametrization.detach(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [CavityParametrization.detach(v) for v in obj]
        val = getval(obj)
        if isinstance(val, np.ndarray):
            return np.array(val)
        return val

    def theta_to_serializable(self, theta) -> dict:
        t = self.detach(theta)
        return {k: np.asarray(v).tolist() for k, v in t.items()}

    @staticmethod
    def theta_from_serializable(d: dict) -> dict:
        return {k: np.asarray(v, dtype=float) for k, v in d.items()}
