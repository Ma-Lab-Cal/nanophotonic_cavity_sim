"""Recompute perf from saved HDF5 in one convention; run+save a candidate sim.

The four prior designs (design_1, invDes(design_1), seed, invDes(seed)) already
have saved verification HDF5 on disk, so their metrics recompute with **zero
cloud cost**. The subtlety: ``Cavity_simulation.from_saved_simulation_file``
eagerly runs ``full_analysis`` with the *narrow* default resonance window and
caches directional-Q / mode-volume at that resonant frequency. We instead load
without that step and let ``diagnostics.perf_from_simulation`` drive every metric
off the *wide* (±8 THz) window, so all designs share one convention.
"""

from __future__ import annotations

import json

import tidy3d as td

from inverse_design import config as cfg
from inverse_design import diagnostics
from inverse_design import objective
from inverse_design.optimizer import run_with_retries
from inverse_design.parametrization import CavityParametrization

from publication_ready import paths


def _load_sim(hdf5_path):
    """Reload a Cavity_simulation from HDF5 WITHOUT the narrow-window analysis."""
    from main_code.simulation import Cavity_simulation

    sim_data = td.SimulationData.from_file(str(hdf5_path))
    attrs = sim_data.simulation.attrs
    inst = Cavity_simulation(
        n_cells=json.loads(attrs["n_cells"]),
        parameters=json.loads(attrs["parameters"]),
        context=json.loads(attrs["context"]),
        beam_layout=json.loads(attrs["beam_layout"]),
    )
    inst.sim = sim_data.simulation
    inst.sim_data = sim_data
    inst.sim_center = sim_data.simulation.center
    inst.sim_size = sim_data.simulation.size
    inst.run_time = sim_data.simulation.run_time
    inst._compute_defect_bounds()
    inst.nanobeam_medium = inst._resolve_medium(inst.context["medium"], plot=False)
    inst._analysis = {}                       # ensure wide-window-only analysis
    return inst


def perf_from_hdf5(hdf5_path) -> tuple:
    """(perf dict, sim_obj) recomputed in the wide-window diagnostics convention."""
    sim_obj = _load_sim(hdf5_path)
    perf = diagnostics.perf_from_simulation(sim_obj)
    return perf, sim_obj


def perf_for_label(label: str) -> tuple:
    """perf + sim for one of the four registry designs, or the trimmed final."""
    if label in paths.HDF5_REGISTRY:
        return perf_from_hdf5(paths.HDF5_REGISTRY[label])
    if label == "final":
        return perf_from_hdf5(paths.final_hdf5_path())
    raise KeyError(f"unknown design label '{label}'")


def verify_candidate(layout: dict, par: CavityParametrization, save_name: str,
                     run_time: float, grid_dl=(0.01, 0.01, 0.01),
                     pml_layers: int = None, validate: bool = True) -> tuple:
    """Build, validate, run, and analyse one candidate geometry on the cloud.

    Returns (perf dict, sim_obj). The HDF5 is written to ``VERIFY_DIR/<save>``.
    Raises ValueError on a geometry-validity failure (overlap / rail / gap)
    before spending any credits.
    """
    cfg.ensure_api_key()
    paths.ensure_dirs()

    if validate:
        problems = objective.validate_layout(par, layout)
        if problems:
            raise ValueError(f"invalid layout '{save_name}': {problems}")

    sim_obj = diagnostics.verification_sim_from_layout(
        layout, par, run_time=run_time, grid_size_override=grid_dl)

    if pml_layers is not None:
        bspec = td.BoundarySpec(
            x=td.Boundary.pml(num_layers=pml_layers),
            y=td.Boundary.pml(num_layers=pml_layers),
            z=td.Boundary.pml(num_layers=pml_layers))
        sim_obj.sim = sim_obj.sim.updated_copy(boundary_spec=bspec)

    run_with_retries(
        lambda: sim_obj.run(directory=str(paths.VERIFY_DIR), save_name=save_name),
        label=f"verify:{save_name}")
    sim_obj._analysis = {}
    perf = diagnostics.perf_from_simulation(sim_obj)
    return perf, sim_obj


def _build_frozen_sim(spec, default_run_time, default_grid_dl):
    so = diagnostics.verification_sim_from_layout(
        spec["layout"], spec["par"],
        run_time=spec.get("run_time", default_run_time),
        grid_size_override=spec.get("grid_dl", default_grid_dl))
    if spec.get("pml_layers"):
        n = spec["pml_layers"]
        so.sim = so.sim.updated_copy(boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(num_layers=n), y=td.Boundary.pml(num_layers=n),
            z=td.Boundary.pml(num_layers=n)))
    return so


def verify_batch(specs, default_run_time, default_grid_dl,
                 folder="pub_batch", validate=True) -> dict:
    """Run many independent candidates in ONE cloud batch (parallel wall-clock).

    Each spec: {name, layout, par, run_time?, grid_dl?, pml_layers?}. Reuses any
    {name}.hdf5 already on disk and only batches the missing ones; each result is
    saved as {name}.hdf5 for later reuse. Returns {name: (perf, sim_obj)}.
    """
    cfg.ensure_api_key()
    paths.ensure_dirs()
    results, objs, sims = {}, {}, {}
    for spec in specs:
        name = spec["name"]
        existing = paths.VERIFY_DIR / f"{name}.hdf5"
        if existing.exists():
            try:
                results[name] = perf_from_hdf5(existing)
                print(f"[batch] {name}: reuse")
                continue
            except Exception as exc:                       # diverged/corrupt save
                print(f"[batch] {name}: saved sim unusable ({exc}); re-running")
        if validate:
            problems = objective.validate_layout(spec["par"], spec["layout"])
            if problems:
                raise ValueError(f"invalid layout '{name}': {problems}")
        so = _build_frozen_sim(spec, default_run_time, default_grid_dl)
        objs[name] = so
        sims[name] = so.sim

    if sims:
        batch = td.web.Batch(simulations=sims, folder_name=folder, verbose=True)
        batch_data = batch.run(path_dir=str(paths.VERIFY_DIR / folder))
        # 1) persist every result first, so one bad sim cannot lose the others
        for name, so in objs.items():
            try:
                so.sim_data = batch_data[name]
                so.sim_data.to_file(str(paths.VERIFY_DIR / f"{name}.hdf5"))
            except Exception as exc:
                print(f"[batch] {name}: download/save failed ({exc})")
        # 2) analyse each independently; a diverged sim yields None, not a crash
        for name, so in objs.items():
            try:
                so._analysis = {}
                results[name] = (diagnostics.perf_from_simulation(so), so)
            except Exception as exc:
                print(f"[batch] {name}: analysis failed ({exc}); marking None")
                results[name] = (None, so)
    return results


def verify_or_reuse(layout: dict, par: CavityParametrization, save_name: str,
                    run_time: float, grid_dl=(0.01, 0.01, 0.01),
                    pml_layers: int = None, force: bool = False) -> tuple:
    """Run a candidate, or reuse its saved HDF5 if present (idempotent sweeps).

    Returns (perf, sim_obj, newly_run: bool). Lets the campaign re-run after an
    interruption without re-spending credits on already-completed points.
    """
    existing = paths.VERIFY_DIR / f"{save_name}.hdf5"
    if existing.exists() and not force:
        print(f"[reuse] {save_name}: reusing saved sim")
        perf, sim_obj = perf_from_hdf5(existing)
        return perf, sim_obj, False
    perf, sim_obj = verify_candidate(layout, par, save_name, run_time,
                                     grid_dl=grid_dl, pml_layers=pml_layers)
    return perf, sim_obj, True
