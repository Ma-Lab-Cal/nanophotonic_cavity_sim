"""Automated inverse design for suspended SiN photonic-crystal nanobeam cavities.

This package adds a Tidy3D-autograd-based inverse-design workflow on top of the
existing hand-design pipeline in ``main_code``. It optimizes, simultaneously:

* per-cell lattice constants (hole spacings),
* the hole placement schedule (positions are the cumulative sum of the free
  per-cell spacings),
* closed-spline hole shapes (periodic cubic B-spline control points).

The differentiable path (``parametrization`` -> ``builder`` -> ``objective``)
uses frequency-domain monitors and ``tidy3d.web.run`` adjoint gradients. The
existing time-domain analysis in ``main_code.simulation`` is reused untouched,
through ``diagnostics.FrozenGeometryCavitySimulation``, as the non-gradient
verification path.

Entry point: ``python -m inverse_design.run_optimization --mode {smoke,full}``.
"""
