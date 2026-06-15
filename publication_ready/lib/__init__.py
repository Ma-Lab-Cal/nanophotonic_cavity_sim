"""Shared helpers for the publication-readiness campaign.

These modules add the only new physics-adjacent code in the package:

* ``layout_ops`` — pure-numpy transforms on detached layout dicts
  (global/defect/output scaling, fabrication perturbation),
* ``reuse``      — recompute perf from saved HDF5 in one convention; run+save
  a candidate verification sim,
* ``fields_post``— atom-position g/C/η maps and the loss-budget decomposition,
* ``datasets``   — machine-readable csv/json/npz writers with fixed schemas,
* ``plots``      — the three plotters missing from inverse_design.visualization.

Nothing here modifies ``inverse_design`` or ``main_code``; they are imported.
"""
