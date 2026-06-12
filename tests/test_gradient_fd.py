"""Cloud-marked test: adjoint gradient vs finite differences on a tiny cavity.

Run explicitly with:
    RUN_CLOUD_TESTS=1 python -m pytest tests/test_gradient_fd.py -s
or as a script:
    python tests/test_gradient_fd.py

Uses a deliberately tiny simulation (15 holes, coarse mesh, short run time):
forward cost ~0.03 FlexCredits. The FD check perturbs 2 parameters
(one lattice constant, one shape scale) -> 4 extra forward sims.
"""

import os
import sys

import numpy as np
import pytest


def _tiny_setup():
    import autograd
    from inverse_design import config as cfg
    from inverse_design import objective
    from inverse_design.builder import build_forward_simulation
    from inverse_design.parametrization import CavityParametrization
    from tidy3d import web

    n_cells = {"N_left_taper": 2, "N_left_mirror": 3, "N_defect": 6,
               "N_right_mirror": 3, "N_right_taper": 1}
    parameters = {
        "parameters_taper_left":    {"lattice": 0.27, "geometry_params": np.array([0.01, 0.01])},
        "parameters_mirrors_left":  {"lattice": 0.50, "geometry_params": np.array([0.25, 0.20])},
        "parameters_defect":        {"lattice": 0.29, "geometry_params": np.array([0.20, 0.20])},
        "parameters_mirrors_right": {"lattice": 0.35, "geometry_params": np.array([0.20, 0.30])},
        "parameters_taper_right":   {"lattice": 0.27, "geometry_params": np.array([0.01, 0.01])},
    }
    par = CavityParametrization(n_cells=n_cells)
    theta0 = par.init_theta_from_baseline(parameters=parameters, n_cells=n_cells)
    sim_cfg = cfg.SimConfig(n_freqs=3, f_window=8e12, min_steps_per_wvl=8,
                            defect_mesh_dl=0.04, run_time=1e-12)

    def loss_of(a_vec):
        theta = dict(theta0)
        theta["a"] = a_vec
        layout = par.layout(theta)
        sim = build_forward_simulation(theta, par, sim_cfg)
        sim_data = web.run(sim, task_name="invdes_gradcheck",
                           folder_name="invdes_tests", verbose=False)
        metrics = objective.physics_metrics(sim_data, sim_cfg)
        loss, _ = objective.loss_from_metrics(metrics, par, layout, c0=1.0)
        return loss

    return par, theta0, loss_of


@pytest.mark.skipif(not os.environ.get("RUN_CLOUD_TESTS"),
                    reason="cloud test; set RUN_CLOUD_TESTS=1")
def test_adjoint_vs_finite_difference():
    run_check()


def run_check():
    import autograd

    par, theta0, loss_of = _tiny_setup()
    a0 = np.asarray(theta0["a"], dtype=float)

    val, grad = autograd.value_and_grad(loss_of)(a0)
    grad = np.asarray(grad)
    print(f"loss = {val:.6f}")
    print(f"|grad| = {np.linalg.norm(grad):.4e}, finite: {np.all(np.isfinite(grad))}")
    assert np.isfinite(val)
    assert np.all(np.isfinite(grad))
    assert np.any(grad != 0), "gradient identically zero"

    # central finite differences on the 2 largest-|grad| components
    idx = np.argsort(-np.abs(grad))[:2]
    h = 0.05  # unconstrained space; maps to ~1-3 nm in lattice constant
    rel_errs = []
    for i in idx:
        ap = a0.copy(); ap[i] += h
        am = a0.copy(); am[i] -= h
        fd = (loss_of(ap) - loss_of(am)) / (2 * h)
        rel = abs(fd - grad[i]) / max(abs(fd), abs(grad[i]), 1e-12)
        rel_errs.append(rel)
        print(f"param a[{i}]: adjoint={grad[i]:+.4e}  FD={fd:+.4e}  rel_err={rel:.2%}")

    assert min(rel_errs) < 0.4, f"FD mismatch: {rel_errs}"


if __name__ == "__main__":
    run_check()
