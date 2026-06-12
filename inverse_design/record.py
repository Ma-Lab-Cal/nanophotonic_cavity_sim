"""Optimization record: serialization and reconstruction.

The record is a list of dictionaries, one per iteration, with keys:

* ``params`` — everything needed to reconstruct the device at that iteration
  (unconstrained theta, the derived physical arrays, cell counts, context,
  and the simulation-config snapshot including the frequency window),
* ``loss``   — scalar loss value,
* ``time``   — ISO-8601 timestamp plus the iteration wall-clock duration (s),
* ``perf``   — physical metrics: differentiable proxies every iteration, and
  the full time-domain diagnostics on diagnostic iterations.

Saved as both ``record.json`` (portable) and ``record.pkl`` (fast).
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np

from inverse_design.parametrization import CavityParametrization


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def make_record_entry(par: CavityParametrization, theta, loss, duration_s,
                      timestamp_iso: str, sim_cfg, perf: dict,
                      stage: str, iteration: int) -> dict:
    layout_np = CavityParametrization.detach(par.layout(theta))
    params = {
        "iteration": iteration,
        "stage": stage,
        "theta": par.theta_to_serializable(theta),
        "physical": {
            "positions": layout_np["positions"],
            "a": layout_np["a"],
            "sx": layout_np["sx"],
            "sy": layout_np["sy"],
            "rho_holes": layout_np["rho_holes"],
        },
        "n_cells": dict(par.n_cells),
        "context": dict(par.context),
        "sim_config": asdict(sim_cfg),
    }
    return {
        "params": _to_jsonable(params),
        "loss": float(loss),
        "time": {"timestamp": timestamp_iso, "duration_s": float(duration_s)},
        "perf": _to_jsonable(perf),
    }


def save_record(record: list, out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "record.pkl", "wb") as f:
        pickle.dump(record, f)
    with open(out_dir / "record.json", "w") as f:
        json.dump(record, f, indent=1)


def load_record(out_dir: Path) -> list:
    out_dir = Path(out_dir)
    pkl = out_dir / "record.pkl"
    if pkl.exists():
        with open(pkl, "rb") as f:
            return pickle.load(f)
    with open(out_dir / "record.json") as f:
        return json.load(f)


def theta_from_entry(entry: dict) -> dict:
    return CavityParametrization.theta_from_serializable(entry["params"]["theta"])


def reconstruct_layout(entry: dict) -> dict:
    """Rebuild the (detached) physical layout from a record entry and verify
    it matches the stored physical arrays."""
    par = CavityParametrization(n_cells=entry["params"]["n_cells"],
                                context=entry["params"]["context"])
    theta = theta_from_entry(entry)
    layout = CavityParametrization.detach(par.layout(theta))
    stored = entry["params"]["physical"]
    for key in ("positions", "a", "sx", "sy"):
        if not np.allclose(layout[key], np.asarray(stored[key]), atol=1e-9):
            raise ValueError(f"record reconstruction mismatch in '{key}'")
    return layout
