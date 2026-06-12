import autograd
import autograd.numpy as anp
import numpy as np

from inverse_design import config as cfg
from inverse_design import objective
from inverse_design.parametrization import CavityParametrization


def _baseline():
    par = CavityParametrization()
    theta0 = par.init_theta_from_baseline()
    return par, theta0


def test_baseline_nearly_penalty_free():
    par, theta0 = _baseline()
    layout = par.layout(theta0)
    pens = objective.penalties(par, layout, curvature=True)
    pens = {k: float(np.asarray(CavityParametrization.detach(v)))
            for k, v in pens.items()}
    # barriers should be essentially inactive at the baseline
    assert pens["gap"] < 0.5, pens
    assert pens["rail"] < 0.5, pens
    assert pens["curv"] < 0.5, pens


def test_gap_penalty_fires_on_overlap():
    par, theta0 = _baseline()
    base = float(np.asarray(CavityParametrization.detach(
        objective.penalties(par, par.layout(theta0))["gap"])))
    # shrink all lattice constants to near the lower bound -> holes collide
    theta_bad = dict(theta0)
    theta_bad["a"] = theta0["a"] - 10.0   # sigmoid -> near lower bound 0.2 um
    bad = float(np.asarray(CavityParametrization.detach(
        objective.penalties(par, par.layout(theta_bad))["gap"])))
    assert bad > 10 * max(base, 1e-6)


def test_rail_penalty_fires_on_wide_holes():
    par, theta0 = _baseline()
    theta_bad = dict(theta0)
    theta_bad["sy"] = theta0["sy"] + 10.0  # sy -> near upper bound 0.42 um
    bad = float(np.asarray(CavityParametrization.detach(
        objective.penalties(par, par.layout(theta_bad))["rail"])))
    assert bad > 1.0


def test_penalty_gradients_finite():
    par, theta0 = _baseline()

    def total_penalty(a):
        theta = dict(theta0)
        theta["a"] = a
        pens = objective.penalties(par, par.layout(theta), curvature=True)
        return sum(pens.values())

    g = autograd.grad(total_penalty)(theta0["a"])
    assert np.all(np.isfinite(g))


def test_validate_layout_baseline_ok():
    par, theta0 = _baseline()
    problems = objective.validate_layout(par, par.layout(theta0))
    assert problems == []


def test_validate_layout_catches_overlap():
    par, theta0 = _baseline()
    theta_bad = dict(theta0)
    theta_bad["a"] = theta0["a"] - 10.0
    theta_bad["sx"] = theta0["sx"] + 10.0
    problems = objective.validate_layout(par, par.layout(theta_bad))
    assert problems != []
