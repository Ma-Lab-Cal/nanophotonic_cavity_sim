import autograd
import autograd.numpy as anp
import numpy as np
import pytest

from inverse_design import config as cfg
from inverse_design.parametrization import (
    CavityParametrization, expand_symmetric_rho, inverse_sigmoid_bounded,
    make_anchor_weights, make_periodic_bspline_basis, sigmoid_bounded)


def test_basis_partition_of_unity():
    B = make_periodic_bspline_basis()
    assert B.shape == (cfg.N_SHAPE_PTS, cfg.N_CTRL)
    assert np.allclose(B.sum(axis=1), 1.0)
    # constant control points -> constant radius
    r = B @ np.full(cfg.N_CTRL, 0.7)
    assert np.allclose(r, 0.7)


def test_sigmoid_round_trip():
    v = np.array([0.21, 0.3, 0.55])
    x = inverse_sigmoid_bounded(v, 0.2, 0.6)
    assert np.allclose(sigmoid_bounded(x, 0.2, 0.6), v, atol=1e-12)


def test_anchor_weights_shape_and_sum():
    W = make_anchor_weights(cfg.BASELINE_N_CELLS)
    assert W.shape == (86, cfg.N_ANCHORS)
    assert np.allclose(W.sum(axis=1), 1.0)
    assert np.all(W >= 0)


def test_baseline_layout_reproduction():
    """theta0 must reproduce the legacy Cavity beam_layout exactly."""
    from main_code.cavity import Cavity

    par = CavityParametrization()
    theta0 = par.init_theta_from_baseline()
    layout = CavityParametrization.detach(par.layout(theta0))

    cav = Cavity(n_cells=dict(cfg.BASELINE_N_CELLS),
                 parameters=cfg.BASELINE_PARAMETERS,
                 context=dict(cfg.BASELINE_CONTEXT))
    legacy = cav.beam_layout

    assert np.allclose(layout["a"], legacy["lattice"], atol=1e-9)
    gp = np.atleast_2d(legacy["geometry_params"])
    assert np.allclose(layout["sx"], gp[:, 0], atol=1e-9)
    assert np.allclose(layout["sy"], gp[:, 1], atol=1e-9)
    # positions agree up to a single global shift
    shift = layout["positions"] - np.asarray(legacy["positions"])
    assert np.allclose(shift, shift[0], atol=1e-9)


def test_baseline_holes_are_ellipses():
    par = CavityParametrization()
    theta0 = par.init_theta_from_baseline()
    layout = CavityParametrization.detach(par.layout(theta0))
    # defect hole (rho=1) should be an exact ellipse with diameters (sx, sy)
    i = 30
    v = np.asarray(layout["vertices"][i])
    rx, ry = layout["sx"][i] / 2, layout["sy"][i] / 2
    ellipse_r = np.sqrt((v[:, 0] / rx) ** 2 + (v[:, 1] / ry) ** 2)
    assert np.allclose(ellipse_r, 1.0, atol=1e-9)


def test_polygons_closed_and_valid():
    shapely = pytest.importorskip("shapely")
    from shapely.geometry import Polygon

    par = CavityParametrization()
    theta0 = par.init_theta_from_baseline()
    # perturb the shape control points significantly
    rng = np.random.default_rng(0)
    theta0["rho"] = theta0["rho"] + rng.normal(0, 2.0, theta0["rho"].shape)
    layout = CavityParametrization.detach(par.layout(theta0))
    for v in layout["vertices"]:
        p = Polygon(np.asarray(v))
        assert p.is_valid
        assert p.area > 0


def test_symmetry_expansion():
    rho_free = np.array([1.0, 2.0, 3.0, 4.0])
    full = expand_symmetric_rho(rho_free)
    assert full.shape == (cfg.N_CTRL,)
    # x-mirror: rho_k == rho_{N-k}
    for k in range(1, cfg.N_CTRL):
        assert full[k] == full[(cfg.N_CTRL - k) % cfg.N_CTRL]
    # y-mirror: rho_k == rho_{(6-k) mod N}
    for k in range(cfg.N_CTRL):
        assert full[k] == full[(6 - k) % cfg.N_CTRL]


def test_layout_gradients_finite():
    par = CavityParametrization()
    theta0 = par.init_theta_from_baseline()

    def scalar_of_layout(rho):
        theta = dict(theta0)
        theta["rho"] = rho
        layout = par.layout(theta)
        total = 0.0
        for v in layout["vertices"]:
            total = total + anp.sum(v ** 2)
        return total + anp.sum(layout["positions"] ** 2)

    g = autograd.grad(scalar_of_layout)(theta0["rho"])
    assert np.all(np.isfinite(g))
    assert np.any(g != 0)


def test_serialization_round_trip():
    par = CavityParametrization()
    theta0 = par.init_theta_from_baseline()
    ser = par.theta_to_serializable(theta0)
    theta1 = CavityParametrization.theta_from_serializable(ser)
    for k in theta0:
        assert np.allclose(theta0[k], theta1[k], atol=1e-15)
