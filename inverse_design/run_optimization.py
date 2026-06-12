"""Inverse-design entry point.

Usage (from the repo root, inside the tidyEnv conda environment):

    python -m inverse_design.run_optimization --mode smoke
    python -m inverse_design.run_optimization --mode full
    python -m inverse_design.run_optimization --mode full --resume
    python -m inverse_design.run_optimization --mode full --skip-baseline

Stages:
  0. Baseline calibration — run (or reuse) the design_1 verification sim,
     compute its honest perf metrics, and calibrate the cooperativity proxy.
  A/B. Gradient optimization (layout, then +spline shapes).
  C. Final full verification + all required plots / GIF / GDS / record.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from inverse_design import config as cfg
from inverse_design import record as record_mod
from inverse_design.optimizer import InverseDesigner, run_with_retries
from inverse_design.parametrization import CavityParametrization


def evaluate_baseline(out_dir: Path, force=False, par=None,
                      seed_parameters=None) -> dict:
    """Stage 0: design_1 baseline perf, seed perf (if seeded), and the proxy
    calibration constant c0 (calibrated on whatever design theta0 encodes)."""
    from tidy3d import web
    from inverse_design import diagnostics, objective
    from inverse_design.builder import build_forward_simulation

    cache = out_dir / "baseline.json"
    if cache.exists() and not force:
        with open(cache) as f:
            return json.load(f)

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) time-domain verification of design_1 (existing pipeline)
    print("[stage 0] building baseline (design_1) verification simulation...")
    sim_obj = diagnostics.baseline_verification_sim()

    def _usable(path):
        """Previously saved baseline run with a non-empty late-time monitor."""
        import tidy3d as td
        try:
            sd = td.SimulationData.from_file(str(path))
            return sd if sd["Field_Time_Monitor"].Ex.sizes["t"] > 0 else None
        except Exception:
            return None

    reused = None
    if not force:
        for cand in (out_dir / "verification" / "baseline_design1.hdf5",
                     Path("cavities_780/design_1.hdf5")):
            if cand.exists():
                reused = _usable(cand)
                if reused is not None:
                    print(f"[stage 0] reusing {cand}")
                    break
    if reused is not None:
        sim_obj.sim_data = reused
        sim_obj.sim = reused.simulation
    else:
        run_with_retries(
            lambda: sim_obj.run(directory=str(out_dir / "verification"),
                                save_name="baseline_design1"),
            label="baseline verification")
    baseline_perf = diagnostics.perf_from_simulation(sim_obj)
    print("[stage 0] baseline perf:", json.dumps(baseline_perf, indent=1))

    # 1b) seed verification, if optimizing from a non-design_1 seed
    seed_perf = None
    if seed_parameters is not None:
        from main_code.cavity import Cavity
        print("[stage 0] verifying the seed design...")
        cav = Cavity(n_cells=dict(par.n_cells), parameters=seed_parameters,
                     context=dict(cfg.BASELINE_CONTEXT))
        cav.build_simulation(grid_size_override=(0.01, 0.01, 0.01),
                             num_modes=1, plot=False)
        seed_sim = cav.simulation
        seed_sim.sim = seed_sim.sim.updated_copy(shutoff=0.0)
        run_with_retries(
            lambda: seed_sim.run(directory=str(out_dir / "verification"),
                                 save_name="seed_design"),
            label="seed verification")
        seed_perf = diagnostics.perf_from_simulation(seed_sim)
        print("[stage 0] seed perf:", json.dumps(seed_perf, indent=1))

    # 2) forward frequency-domain sim of theta0 -> calibration c0
    # (theta0 encodes the seed when one is given, else design_1)
    print("[stage 0] running forward (frequency-domain) sim of theta0...")
    par = par or CavityParametrization()
    theta0 = par.init_theta_from_baseline(parameters=seed_parameters,
                                          n_cells=par.n_cells)
    calib_perf = seed_perf if seed_perf is not None else baseline_perf
    sim_cfg = cfg.SimConfig(f_center=calib_perf["f_res_Hz"], f_window=2e12)
    sim = build_forward_simulation(theta0, par, sim_cfg)
    sim_data = run_with_retries(
        lambda: web.run(sim, task_name="invdes_stage0_theta0",
                        folder_name=out_dir.name, verbose=False),
        label="stage0 forward")
    metrics = objective.physics_metrics(sim_data, sim_cfg)
    C_raw0 = float(np.asarray(metrics["C_raw"]))
    beta0 = float(np.asarray(metrics["beta"]))
    c0 = calib_perf["C_atom"] / max(C_raw0, 1e-30)

    result = {"baseline_perf": baseline_perf, "seed_perf": seed_perf,
              "C_raw0": C_raw0, "beta0": beta0, "c0": c0,
              "f_hat0": float(np.asarray(metrics["f_hat"]))}
    with open(cache, "w") as f:
        json.dump(result, f, indent=1)
    print(f"[stage 0] c0 = {c0:.4e}, beta0 = {beta0:.3f}, "
          f"f_hat0 = {result['f_hat0'] / 1e12:.2f} THz")
    return result


def finalize(designer: InverseDesigner, out_dir: Path):
    """Stage C: final verification + all required outputs."""
    from inverse_design import diagnostics, visualization

    rec = designer.record
    if not rec:
        print("[finalize] empty record; nothing to plot")
        return

    final_sim, final_sim_data = None, None
    try:
        layout_np = CavityParametrization.detach(
            designer.par.layout(designer.theta))
        sim_obj = diagnostics.verification_sim_from_layout(layout_np, designer.par)
        final_sim = sim_obj.sim
        run_with_retries(
            lambda: sim_obj.run(directory=str(out_dir / "verification"),
                                save_name="final_design"),
            label="final verification")
        final_sim_data = sim_obj.sim_data
        final_perf = diagnostics.perf_from_simulation(sim_obj)
        rec[-1]["perf"]["diagnostic"] = record_mod._to_jsonable(final_perf)
        record_mod.save_record(rec, out_dir)
        print("[finalize] final perf:", json.dumps(
            record_mod._to_jsonable(final_perf), indent=1))
    except Exception as exc:
        print(f"[finalize] final verification failed/skipped: {exc}")

    visualization.generate_all(
        rec, designer.baseline_perf, out_dir, final_sim=final_sim,
        final_sim_data=final_sim_data,
        total_time_s=designer.total_opt_time_s)
    print(f"[finalize] outputs written to {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Nanobeam cavity inverse design")
    ap.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-baseline", action="store_true",
                    help="reuse cached baseline.json without recomputing")
    ap.add_argument("--credit-cap", type=float, default=None)
    ap.add_argument("--iters-a", type=int, default=None,
                    help="override Stage A iteration count (full mode)")
    ap.add_argument("--iters-b", type=int, default=None,
                    help="override Stage B iteration count (full mode)")
    ap.add_argument("--seed-file", default=None,
                    help="JSON file with {'n_cells', 'parameters'} (legacy "
                         "section format) to seed the optimization from "
                         "instead of design_1")
    args = ap.parse_args()

    cfg.ensure_api_key()

    run_cfg = (cfg.smoke_run_config(args.run_name or "smoke")
               if args.mode == "smoke"
               else cfg.full_run_config(args.run_name or "full"))
    if args.credit_cap is not None:
        run_cfg.credit_cap = args.credit_cap
    if args.mode == "full":
        if args.iters_a is not None:
            run_cfg.stages[0].n_iters = args.iters_a
        if args.iters_b is not None:
            run_cfg.stages[1].n_iters = args.iters_b
    out_dir = run_cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_parameters, par = None, None
    if args.seed_file:
        with open(args.seed_file) as f:
            seed = json.load(f)
        seed_parameters = {
            k: {"lattice": float(v["lattice"]),
                "geometry_params": np.asarray(v["geometry_params"], dtype=float)}
            for k, v in seed["parameters"].items()}
        par = CavityParametrization(n_cells=seed["n_cells"])
        print(f"[seed] optimizing from {args.seed_file} "
              f"({sum(seed['n_cells'].values())} holes)")

    t0 = time.time()
    stage0 = evaluate_baseline(out_dir, force=False, par=par,
                               seed_parameters=seed_parameters)
    designer = InverseDesigner(run_cfg, par=par, c0=stage0["c0"],
                               baseline_perf=stage0["baseline_perf"],
                               init_parameters=seed_parameters)
    calib = stage0.get("seed_perf") or stage0["baseline_perf"]
    designer.f_center = calib["f_res_Hz"]

    if args.resume:
        designer.resume()

    try:
        designer.run()
    except KeyboardInterrupt:
        print("[interrupted] checkpointing...")
        designer.checkpoint()

    finalize(designer, out_dir)
    print(f"[total] wall time {time.time() - t0:.0f} s, "
          f"optimization time {designer.total_opt_time_s:.0f} s, "
          f"credits ~{designer.credits_spent:.1f}")


if __name__ == "__main__":
    main()
