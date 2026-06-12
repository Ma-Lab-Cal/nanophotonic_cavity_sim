import numpy as np
import pytest

from inverse_design import config as cfg
from inverse_design.parametrization import CavityParametrization


@pytest.fixture(scope="module")
def par_theta():
    par = CavityParametrization()
    theta0 = par.init_theta_from_baseline()
    return par, theta0


def test_build_structures_counts(par_theta):
    import tidy3d as td
    from inverse_design.builder import build_structures

    par, theta0 = par_theta
    layout = par.layout(theta0)
    structures = build_structures(layout, par, 50.0)
    assert len(structures) == 2
    slab, holes = structures
    assert isinstance(holes.geometry, td.GeometryGroup)
    assert len(holes.geometry.geometries) == par.n_holes
    assert float(holes.medium.permittivity) == 1.0


def test_forward_simulation_validates(par_theta):
    from inverse_design.builder import build_forward_simulation

    par, theta0 = par_theta
    sim_cfg = cfg.SimConfig(n_freqs=5, min_steps_per_wvl=8, defect_mesh_dl=0.05)
    sim = build_forward_simulation(theta0, par, sim_cfg)

    names = {m.name for m in sim.monitors}
    expected = {"wg_out", "atom_0", "atom_1", "atom_2", "profile_z",
                "box_+x", "box_-x", "box_+y", "box_-y", "box_+z", "box_-z"}
    assert expected <= names
    assert sim.symmetry == (0, -1, 1)

    # monitor frequency window centered correctly
    wg = [m for m in sim.monitors if m.name == "wg_out"][0]
    assert np.isclose(np.mean(wg.freqs), sim_cfg.f_center)

    # full local validation (pydantic + pre-upload checks)
    sim.validate_pre_upload()


def test_record_entry_and_reconstruction(par_theta, tmp_path):
    from inverse_design import record as record_mod

    par, theta0 = par_theta
    sim_cfg = cfg.SimConfig()
    entry = record_mod.make_record_entry(
        par, theta0, loss=1.23, duration_s=10.0,
        timestamp_iso="2026-06-11T00:00:00", sim_cfg=sim_cfg,
        perf={"proxy": {"eta_tilde": 0.5}}, stage="test", iteration=0)

    assert set(entry.keys()) == {"params", "loss", "time", "perf"}
    record_mod.save_record([entry], tmp_path)
    loaded = record_mod.load_record(tmp_path)
    assert loaded[0]["loss"] == 1.23

    layout = record_mod.reconstruct_layout(loaded[0])
    assert np.all(np.isfinite(layout["positions"]))


def test_visualization_smoke(par_theta, tmp_path):
    """Plots + GIF from a tiny synthetic record (no cloud)."""
    from inverse_design import record as record_mod
    from inverse_design import visualization

    par, theta0 = par_theta
    sim_cfg = cfg.SimConfig()
    rec = []
    for i in range(3):
        theta = {k: np.asarray(v) + 0.01 * i for k, v in theta0.items()}
        perf = {"proxy": {"eta_tilde": 0.1 + 0.1 * i, "C_tilde": 1.0,
                          "beta": 0.5, "C_raw": 1.0, "f_hat": 384e12,
                          "L_phys": 2.0 - i, "L_freq": 0.1}}
        if i == 2:
            perf["diagnostic"] = {
                "f_res_Hz": 384.2e12, "Q": 1e5, "beta_wg": 0.7,
                "Vmode_lambda_n3": 0.5, "purcell": 1e4,
                "g_2pi_Hz_at_250nm": 5e8, "C_atom": 30.0, "eta_atom": 0.65}
        rec.append(record_mod.make_record_entry(
            par, theta, loss=2.0 - i, duration_s=5.0,
            timestamp_iso=f"2026-06-11T00:0{i}:00", sim_cfg=sim_cfg,
            perf=perf, stage="test", iteration=i))

    baseline_perf = {"f_res_Hz": 389.96e12, "Q": 1.33e6, "beta_wg": 0.3,
                     "Vmode_lambda_n3": 0.18, "purcell": 5.6e5,
                     "g_2pi_Hz_at_250nm": 4e8, "C_atom": 20.0, "eta_atom": 0.4}
    visualization.generate_all(rec, baseline_perf, tmp_path)

    assert (tmp_path / "loss_history.png").exists()
    assert (tmp_path / "perf_history.png").exists()
    assert (tmp_path / "cross_sections_final.png").exists()
    assert (tmp_path / "structure_evolution.gif").exists()
