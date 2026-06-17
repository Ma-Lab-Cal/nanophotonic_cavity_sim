"""Configuration for the inverse-design workflow.

Holds the frozen baseline ("current best design": design_1 from
``cavities_780.ipynb``, re-expressed in the current ``geometry_params`` API),
physical constants for the Rb D2 interface, and the run/stage configuration
dataclasses used by the optimizer driver.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "invDesResults"

# ── Physics constants (Rb D2) ─────────────────────────────────────────────
C0_UM = 299792458.0 * 1e6        # speed of light (um/s), matches td.constants.C_0
F_TARGET = C0_UM / 0.780241      # 87Rb D2 line, 780.241 nm -> ~384.23 THz (Hz)
DIPOLE_MOMENT = 3.58e-29         # C*m, Rb D2 <J=1/2||er||J'=3/2> reduced dipole matrix element (Steck)
GAMMA_RB = 2 * np.pi * 6.0666e6  # rad/s, Rb D2 natural linewidth (gamma = 2pi*6.07 MHz)
FIBER_EFFICIENCY = 0.99          # eta_fiber used in eta(r); same default as cooperativity()

# Atom sits in the tweezer retro-reflection antinode above the beam surface.
ATOM_SURFACE_OFFSETS_UM = (0.20, 0.25, 0.30)   # offsets above the top surface (z = thickness/2)
ATOM_Z_WEIGHTS = (0.25, 0.5, 0.25)             # loss weighting over the three heights

# ── Baseline: design_1 from cavities_780.ipynb (current best 780 nm design) ──
# The notebook used the legacy "hole_params" key; the current main_code API is
# "geometry_params". Values are identical.
BASELINE_N_CELLS = {
    "N_left_taper": 5,
    "N_left_mirror": 10,
    "N_defect": 40,
    "N_right_mirror": 30,
    "N_right_taper": 1,
}

BASELINE_PARAMETERS = {
    "parameters_taper_left":    {"lattice": 0.27, "geometry_params": np.array([0.01, 0.01])},
    "parameters_mirrors_left":  {"lattice": 0.50, "geometry_params": np.array([0.25, 0.20])},
    "parameters_defect":        {"lattice": 0.29, "geometry_params": np.array([0.20, 0.20])},
    "parameters_mirrors_right": {"lattice": 0.35, "geometry_params": np.array([0.20, 0.30])},
    "parameters_taper_right":   {"lattice": 0.27, "geometry_params": np.array([0.01, 0.01])},
}

BASELINE_CONTEXT = {
    "freq0": F_TARGET,
    "fwidth": F_TARGET * 0.01,
    "thickness": 0.15,
    "width": 0.5,
    "polarization": "Ey",
    "medium": str(REPO_ROOT / "materials" / "SiN.txt"),
    "mode": "dielectric",
    "sidewall_angle": 0,
    "geometry": "ellipse",
}

# Output port: design_1 has 10 left mirrors vs 30 right mirrors -> photons
# leave through the LEFT (-x) waveguide.
OUTPUT_DIRECTION = "-"           # ModeMonitor direction for the output port
OUTPUT_FACE = "-x"               # directional-Q face name of the output port

# ── Parameter bounds (physical units, um) ─────────────────────────────────
A_BOUNDS = (0.20, 0.60)          # per-cell lattice constant
# Lower bound 8 nm: the legacy taper tips use 10 nm holes, and letting holes
# shrink to sub-fabrication size is how the optimizer can effectively REMOVE
# a hole (the hole-frequency knob). Holes below ~20 nm are treated as absent
# at fabrication time.
SX_BOUNDS = (0.008, 0.55)        # hole x-diameter scale at rho=1
SY_BOUNDS = (0.008, 0.42)        # hole y-diameter scale at rho=1 (width 0.5 - 2*rail)
RHO_BOUNDS = (0.5, 1.5)          # spline control radius ratio

# Spline shape model
N_CTRL = 12                      # control points around the closed spline
N_SHAPE_PTS = 64                 # polygon vertices per hole
N_FREE_RHO = 4                   # free control radii after x+y mirror symmetry
N_ANCHORS = 5                    # shape anchors: taperL, mirrorL, defect, mirrorR, taperR

# Fabrication / validity constraints
GAP_MIN_UM = 0.050               # min dielectric gap between neighboring holes
RAIL_MIN_UM = 0.070              # min side rail (beam edge to hole edge)
PENALTY_SCALE_UM = 0.005         # softplus scale for the constraint barriers
HOLE_VANISH_UM = 0.020           # holes smaller than this count as absent
HOLE_EXISTS_UM = 0.040           # full fabricability constraints above this

# Loss weights
W_FREQ = 1.0                     # resonance-targeting weight
FREQ_TOL_HZ = 0.5e12             # resonance penalty normalization (0.5 THz)
LAMBDA_GAP = 10.0
LAMBDA_RAIL = 10.0
LAMBDA_CURV = 1.0                # spline curvature penalty (stage B)
LAMBDA_SMOOTH = 0.01             # lattice smoothness (gentle confinement prior)
SOFTMAX_TAU = 3.0                # frequency soft-peak-picking temperature


@dataclass
class SimConfig:
    """Settings for one differentiable forward simulation."""
    f_center: float = F_TARGET           # center of the monitor frequency window (Hz)
    f_window: float = 1.0e12             # window width (Hz)
    n_freqs: int = 11
    fwidth_source: float = 4.0e12        # source bandwidth (Hz)
    min_steps_per_wvl: int = 15
    defect_mesh_dl: float = 0.015        # mesh override in the defect region (um)
    run_time: float = 5e-12              # s
    shutoff: float = 1e-6

    @property
    def freqs(self) -> np.ndarray:
        return np.linspace(self.f_center - self.f_window / 2,
                           self.f_center + self.f_window / 2, self.n_freqs)


@dataclass
class StageConfig:
    name: str = "A"
    n_iters: int = 50
    learning_rate: float = 0.03
    lr_final: float = 0.01               # cosine decay target
    optimize_rho: bool = False           # unlock spline control points
    curvature_penalty: bool = False
    diag_every: int = 10                 # run time-domain verification every N iters
    sim: SimConfig = field(default_factory=SimConfig)


@dataclass
class RunConfig:
    run_name: str = "run"
    credit_cap: float = 290.0
    stages: list = field(default_factory=list)

    @property
    def out_dir(self) -> Path:
        return RESULTS_ROOT / self.run_name


def smoke_run_config(run_name: str = "smoke") -> RunConfig:
    sim = SimConfig(f_window=4.0e12, n_freqs=5, min_steps_per_wvl=10,
                    defect_mesh_dl=0.03, run_time=2e-12)
    stage = StageConfig(name="smoke", n_iters=2, learning_rate=0.02,
                        lr_final=0.02, optimize_rho=True,
                        curvature_penalty=True, diag_every=1, sim=sim)
    return RunConfig(run_name=run_name, credit_cap=10.0, stages=[stage])


def full_run_config(run_name: str = "full") -> RunConfig:
    # Forward-sim quality chosen from smoke-run evidence: even mspw=10 proxies
    # tracked the verified diagnostics; mspw=12 / 4 ps is the cost-accuracy
    # sweet spot for optimization, with full-quality verification at each
    # diagnostic iteration and at the end.
    # n_freqs: each objective frequency spawns its own adjoint simulation
    # (multi-monitor objective; no broadband combining), so the frequency
    # count directly multiplies the per-iteration cost and adjoint count.
    # f_window must be WIDE: with a narrow window the softmax peak-pick
    # saturates at the window edge when the resonance drifts, the frequency
    # penalty loses its gradient, and the optimizer happily walks the
    # resonance away from the target (observed in Stage A: 385 -> 374 THz).
    sim = SimConfig(f_window=6.0e12, n_freqs=9, min_steps_per_wvl=12,
                    defect_mesh_dl=0.02, run_time=4e-12)
    stage_a = StageConfig(name="A_layout", n_iters=50, learning_rate=0.03,
                          lr_final=0.01, optimize_rho=False,
                          curvature_penalty=False, diag_every=5, sim=sim)
    stage_b = StageConfig(name="B_shapes", n_iters=50, learning_rate=0.015,
                          lr_final=0.005, optimize_rho=True,
                          curvature_penalty=True, diag_every=5, sim=sim)
    return RunConfig(run_name=run_name, credit_cap=290.0, stages=[stage_a, stage_b])


def ensure_api_key():
    """Make sure the Tidy3D web API is configured.

    Priority: existing ~/.tidy3d/config, then the TIDY3D_API_KEY environment
    variable (which we persist via web.configure). Raises if neither found.
    """
    import tidy3d.web as web

    candidates = (
        Path.home() / ".tidy3d" / "config",                 # legacy location
        Path.home() / ".config" / "tidy3d" / "config.toml",  # tidy3d >= 2.x
    )
    if any(p.exists() for p in candidates):
        return
    key = os.environ.get("TIDY3D_API_KEY")
    if key:
        web.configure(key)
        return
    raise RuntimeError(
        "No Tidy3D API key configured. Either create ~/.tidy3d/config "
        "(e.g. by running tidy3d.web.configure(API_KEY) once) or set the "
        "TIDY3D_API_KEY environment variable."
    )
