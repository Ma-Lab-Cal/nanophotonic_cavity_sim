"""Inverse-design optimization driver.

Adam (from tidy3d.plugins.autograd) over the unconstrained theta, with:

* per-iteration checkpointing (theta + Adam state + frequency window + credit
  ledger) and exact resume,
* NaN/cloud-failure guards (skip update, halve learning rate, retry/backoff),
* a hard FlexCredit ledger that stops the run cleanly at the configured cap,
* periodic non-gradient verification ("diagnostic iterations") through the
  existing time-domain pipeline,
* the reconstruction record (record.py) updated every iteration.
"""

from __future__ import annotations

import datetime
import pickle
import time
import traceback
from pathlib import Path

import autograd
import autograd.numpy as anp
import numpy as np

from inverse_design import config as cfg
from inverse_design import objective, record
from inverse_design.parametrization import CavityParametrization

# theta blocks in a fixed order for flatten/unflatten
_BLOCKS = ("a", "sx", "sy", "rho")


def flatten_theta(theta):
    return np.concatenate([np.asarray(theta[k], dtype=float).ravel() for k in _BLOCKS])


def unflatten_theta(vec, shapes):
    out, i = {}, 0
    for k in _BLOCKS:
        n = int(np.prod(shapes[k]))
        out[k] = anp.reshape(vec[i:i + n], shapes[k])
        i += n
    return out


class AdamState:
    def __init__(self, n, lr):
        self.m = np.zeros(n)
        self.v = np.zeros(n)
        self.t = 0
        self.lr = lr

    def update(self, grad, lr=None):
        b1, b2, eps = 0.9, 0.999, 1e-8
        if lr is not None:
            self.lr = lr
        self.t += 1
        self.m = b1 * self.m + (1 - b1) * grad
        self.v = b2 * self.v + (1 - b2) * grad ** 2
        mhat = self.m / (1 - b1 ** self.t)
        vhat = self.v / (1 - b2 ** self.t)
        return -self.lr * mhat / (np.sqrt(vhat) + eps)


def run_with_retries(fn, n_retries=3, backoff_s=30.0, label="web.run"):
    last_exc = None
    for attempt in range(n_retries):
        try:
            return fn()
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # cloud/API errors
            last_exc = exc
            print(f"[retry] {label} failed (attempt {attempt + 1}/{n_retries}): {exc}")
            traceback.print_exc(limit=2)
            if attempt < n_retries - 1:
                time.sleep(backoff_s * (attempt + 1))
    raise RuntimeError(f"{label} failed after {n_retries} attempts") from last_exc


class InverseDesigner:
    def __init__(self, run_cfg: cfg.RunConfig, par: CavityParametrization = None,
                 c0: float = 1.0, baseline_perf: dict = None, verbose=True):
        self.run_cfg = run_cfg
        self.par = par or CavityParametrization()
        self.c0 = float(c0)
        self.baseline_perf = baseline_perf or {}
        self.verbose = verbose

        self.out_dir = Path(run_cfg.out_dir)
        self.ckpt_dir = self.out_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.theta = self.par.init_theta_from_baseline()
        self.shapes = {k: np.asarray(self.theta[k]).shape for k in _BLOCKS}
        self.record = []
        self.credits_spent = 0.0
        self.global_iter = 0
        self.f_center = cfg.F_TARGET
        self.adam = None
        self.bad_steps = 0
        self.clean_steps = 0
        self.t_run_start = time.time()
        self.total_opt_time_s = 0.0
        self.completed_stages = []

    # ── checkpointing ─────────────────────────────────────────────────

    def checkpoint(self):
        state = {
            "theta": self.par.theta_to_serializable(self.theta),
            "adam": None if self.adam is None else {
                "m": self.adam.m, "v": self.adam.v, "t": self.adam.t,
                "lr": self.adam.lr},
            "credits_spent": self.credits_spent,
            "global_iter": self.global_iter,
            "f_center": self.f_center,
            "c0": self.c0,
            "baseline_perf": self.baseline_perf,
            "total_opt_time_s": self.total_opt_time_s,
            "completed_stages": self.completed_stages,
        }
        with open(self.ckpt_dir / f"iter_{self.global_iter:04d}.pkl", "wb") as f:
            pickle.dump(state, f)
        with open(self.ckpt_dir / "latest.pkl", "wb") as f:
            pickle.dump(state, f)
        record.save_record(self.record, self.out_dir)

    def resume(self, ckpt_path=None):
        path = Path(ckpt_path) if ckpt_path else self.ckpt_dir / "latest.pkl"
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.theta = CavityParametrization.theta_from_serializable(state["theta"])
        if state["adam"] is not None:
            self.adam = AdamState(len(flatten_theta(self.theta)), state["adam"]["lr"])
            self.adam.m = state["adam"]["m"]
            self.adam.v = state["adam"]["v"]
            self.adam.t = state["adam"]["t"]
        self.credits_spent = state["credits_spent"]
        self.global_iter = state["global_iter"]
        self.f_center = state["f_center"]
        self.c0 = state["c0"]
        self.baseline_perf = state.get("baseline_perf", {})
        self.total_opt_time_s = state.get("total_opt_time_s", 0.0)
        self.completed_stages = state.get("completed_stages", [])
        try:
            self.record = record.load_record(self.out_dir)
        except FileNotFoundError:
            self.record = []
        print(f"[resume] from iter {self.global_iter}, "
              f"{self.credits_spent:.1f} credits spent")

    # ── loss function over the flat vector ────────────────────────────

    def _make_loss(self, stage: cfg.StageConfig, sim_cfg: cfg.SimConfig,
                   task_name: str):
        from tidy3d import web
        from inverse_design.builder import build_forward_simulation

        par, c0 = self.par, self.c0
        frozen = {k: np.asarray(self.theta[k]) for k in _BLOCKS}

        def loss_flat(vec):
            theta = unflatten_theta(vec, self.shapes)
            if not stage.optimize_rho:
                theta["rho"] = frozen["rho"]   # constant w.r.t. vec
            layout = par.layout(theta)
            sim = build_forward_simulation(theta, par, sim_cfg)
            # One adjoint simulation is spawned per objective frequency (the
            # multi-monitor objective prevents broadband combining), so the
            # allowance must cover n_freqs.
            sim_data = web.run(sim, task_name=task_name,
                               folder_name=self.run_cfg.run_name,
                               verbose=False, local_gradient=False,
                               max_num_adjoint_per_fwd=16)
            metrics = objective.physics_metrics(sim_data, sim_cfg)
            loss, aux = objective.loss_from_metrics(
                metrics, par, layout, c0, curvature=stage.curvature_penalty)
            return loss, aux

        return loss_flat

    # ── validation gate before any upload ─────────────────────────────

    def _preflight(self, theta, sim_cfg):
        from inverse_design.builder import build_forward_simulation
        layout = self.par.layout(theta)
        problems = objective.validate_layout(self.par, layout)
        if problems:
            return problems
        try:
            sim = build_forward_simulation(theta, self.par, sim_cfg)
            sim.validate_pre_upload()
        except Exception as exc:
            return [f"simulation validation failed: {exc}"]
        return []

    # ── credit accounting ─────────────────────────────────────────────

    def _estimate_iteration_cost(self, sim_cfg) -> float:
        """Estimate forward+adjoint cost of one iteration (FlexCredits)."""
        from tidy3d import web
        from inverse_design.builder import build_forward_simulation
        sim = build_forward_simulation(self.theta, self.par, sim_cfg)
        tid = web.upload(sim, task_name="invdes_cost_probe",
                         folder_name=self.run_cfg.run_name, verbose=False)
        fwd = float(web.estimate_cost(tid, verbose=False))
        try:
            web.delete(tid)
        except Exception:
            pass
        # one adjoint simulation per objective frequency, each ~forward cost
        return (1 + sim_cfg.n_freqs) * fwd

    def _check_budget(self, next_cost) -> bool:
        if self.credits_spent + next_cost > self.run_cfg.credit_cap:
            print(f"[budget] cap {self.run_cfg.credit_cap} would be exceeded "
                  f"({self.credits_spent:.1f} spent + {next_cost:.2f} next); stopping.")
            return False
        return True

    # ── diagnostics (non-gradient) ────────────────────────────────────

    def run_diagnostics(self, theta, label: str, window: float = 6e12) -> dict:
        from inverse_design import diagnostics
        layout_np = CavityParametrization.detach(self.par.layout(theta))
        sim_obj = diagnostics.verification_sim_from_layout(layout_np, self.par)
        run_with_retries(
            lambda: sim_obj.run(directory=str(self.out_dir / "verification"),
                                save_name=label),
            label=f"verification:{label}")
        perf = diagnostics.perf_from_simulation(sim_obj)
        # Adaptive window placement from the verified resonance. The window
        # must contain BOTH the current mode (so the softmax peak-pick sees
        # it) and, if possible, the target (so the frequency penalty has a
        # restoring gradient). With the mode outside the window the penalty
        # saturates and the resonance drifts unchecked.
        f_res = perf["f_res_Hz"]
        if np.isfinite(f_res) and abs(f_res - cfg.F_TARGET) < 30e12:
            delta = cfg.F_TARGET - f_res
            if abs(delta) < 0.8 * window:
                self.f_center = 0.5 * (f_res + cfg.F_TARGET)
            else:
                # window anchored on the mode, extended toward the target
                self.f_center = f_res + 0.35 * window * np.sign(delta)
        return perf

    # ── main loop ─────────────────────────────────────────────────────

    def run_stage(self, stage: cfg.StageConfig):
        sim_cfg = stage.sim
        n = len(flatten_theta(self.theta))
        if self.adam is None:
            self.adam = AdamState(n, stage.learning_rate)

        per_iter_cost = self._estimate_iteration_cost(sim_cfg)
        print(f"[stage {stage.name}] estimated {per_iter_cost:.2f} credits/iter")

        for it in range(stage.n_iters):
            # cosine lr schedule within the stage
            frac = it / max(stage.n_iters - 1, 1)
            lr = stage.lr_final + 0.5 * (stage.learning_rate - stage.lr_final) \
                * (1 + np.cos(np.pi * frac))

            sim_cfg.f_center = self.f_center
            if not self._check_budget(per_iter_cost):
                return False

            problems = self._preflight(self.theta, sim_cfg)
            if problems:
                print(f"[invalid geometry] {problems}; halving lr and skipping")
                self.adam.lr = max(self.adam.lr / 2, 1e-4)
                self.global_iter += 1
                continue

            t0 = time.time()
            ts = datetime.datetime.now().isoformat(timespec="seconds")
            vec = flatten_theta(self.theta)
            loss_flat = self._make_loss(stage, sim_cfg,
                                        f"invdes_{stage.name}_{self.global_iter:04d}")

            try:
                (loss_val, grad), aux = self._value_grad_with_aux(loss_flat, vec)
            except Exception as exc:
                print(f"[iteration failed] {exc}; skipping update")
                self.global_iter += 1
                self.checkpoint()
                continue

            duration = time.time() - t0
            self.total_opt_time_s += duration
            self.credits_spent += per_iter_cost

            grad = np.asarray(grad, dtype=float)
            bad = (not np.isfinite(loss_val)) or (not np.all(np.isfinite(grad)))
            if bad:
                print(f"[NaN guard] non-finite loss/grad at iter {self.global_iter}; "
                      f"skip update, lr/2")
                self.adam.lr = max(self.adam.lr / 2, 1e-4)
                self.bad_steps += 1
                self.clean_steps = 0
            else:
                gnorm = np.linalg.norm(grad)
                if gnorm > 1.0:
                    grad = grad / gnorm
                step = self.adam.update(grad, lr=lr)
                new_vec = vec + step
                self.theta = {k: np.asarray(v) for k, v in
                              unflatten_theta(new_vec, self.shapes).items()}
                self.clean_steps += 1

            # diagnostics every diag_every iterations (and on the last)
            perf = {"proxy": {k: float(np.asarray(
                        CavityParametrization.detach(aux[k])))
                    for k in ("eta_tilde", "C_tilde", "beta", "C_raw", "f_hat",
                              "L_phys", "L_freq")},
                    "credits_spent": self.credits_spent}
            is_diag = (self.global_iter % stage.diag_every == 0
                       or it == stage.n_iters - 1)
            if is_diag and not bad:
                diag_cost = per_iter_cost / 2.2  # one forward-style sim
                if self._check_budget(diag_cost):
                    try:
                        perf["diagnostic"] = self.run_diagnostics(
                            self.theta, f"diag_{self.global_iter:04d}",
                            window=sim_cfg.f_window)
                        self.credits_spent += diag_cost
                    except Exception as exc:
                        print(f"[diagnostics failed] {exc}")

            entry = record.make_record_entry(
                self.par, self.theta, float(loss_val), duration, ts, sim_cfg,
                perf, stage.name, self.global_iter)
            self.record.append(entry)

            if self.verbose:
                d = perf.get("diagnostic", {})
                print(f"iter {self.global_iter:4d} [{stage.name}] "
                      f"loss={loss_val:9.4f} eta~={perf['proxy']['eta_tilde']:.4f} "
                      f"beta={perf['proxy']['beta']:.3f} "
                      f"f^={perf['proxy']['f_hat'] / 1e12:.2f} THz "
                      f"lr={lr:.4f} cred={self.credits_spent:.1f}"
                      + (f" | Q={d.get('Q', 0):.0f} eta={d.get('eta_atom', 0):.3f}"
                         if d else ""))

            self.global_iter += 1
            self.checkpoint()
        return True

    @staticmethod
    def _value_grad_with_aux(loss_flat, vec):
        """value_and_grad that also returns the aux dict from the same run."""
        aux_holder = {}

        def wrapped(v):
            loss, aux = loss_flat(v)
            aux_holder.update(aux)
            return loss

        vg = autograd.value_and_grad(wrapped)
        loss_val, grad = vg(vec)
        return (loss_val, grad), aux_holder

    def run(self):
        ok = True
        for stage in self.run_cfg.stages:
            if stage.name in self.completed_stages:
                print(f"[stage {stage.name}] already completed; skipping")
                continue
            # fresh Adam state per stage (parameter activity changes)
            self.adam = AdamState(len(flatten_theta(self.theta)),
                                  stage.learning_rate)
            ok = self.run_stage(stage)
            if not ok:
                break
            self.completed_stages.append(stage.name)
        self.checkpoint()
        print(f"[done] total optimization time: {self.total_opt_time_s:.0f} s, "
              f"credits: {self.credits_spent:.1f}")
        return ok
