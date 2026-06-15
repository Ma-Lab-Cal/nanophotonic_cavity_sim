"""Mode-isolation scan (postprocess of the final sim; zero cloud).

Re-runs the ResonanceFinder over a broad window (±1 THz around D2) on the final
sim's combined point-monitor signal, and computes the spectrum + the separation
of the cavity mode from the nearest competing resonance (in loaded linewidths).
Writes:
  data/mode_scan.csv          — every resonance in the window
  data/competing_modes.npz    — combined-signal spectrum + selected/competitor

    python -m publication_ready.run_mode_scan
    python -m publication_ready.run_mode_scan --reuse-label invDes(seed)
"""

from __future__ import annotations

import argparse

import numpy as np

from inverse_design import config as cfg

from publication_ready import campaign, paths
from publication_ready.lib import datasets, reuse

F_D2 = cfg.F_TARGET


def _spectrum(sim_obj):
    """|FFT| of the late-time combined point-monitor signal."""
    combined = sim_obj._analysis["combined_signal"]
    start = 2 * sim_obj._find_source_decay()
    sig = combined[combined.t >= start]
    y = np.asarray(sig).real
    n = len(y)
    if n < 8:
        return np.array([]), np.array([])
    dt = float(sim_obj.sim.dt)
    Y = np.fft.rfft(y * np.hanning(n))
    f = np.fft.rfftfreq(n, d=dt)
    return f, np.abs(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse-label", default=None)
    args = ap.parse_args()
    paths.ensure_dirs()

    if args.reuse_label:
        _, sim = reuse.perf_for_label(args.reuse_label)
    else:
        _, sim = reuse.perf_from_hdf5(paths.final_hdf5_path())

    hw = campaign.MODE_SCAN_HALFWINDOW_HZ
    sim.analyse_resonances(freq_window=(F_D2 - hw, F_D2 + hw))
    df = sim.resonance_df
    freqs = np.abs(np.asarray(df.index, dtype=float))
    Qs = np.asarray(df["Q"], dtype=float)
    amps = np.asarray(df["amplitude"], dtype=float)
    errs = np.asarray(df["error"], dtype=float)

    f_sel = float(sim.resonant_frequency)
    kappa_2pi = float(sim.kappa_tot) / (2 * np.pi)        # Hz
    # nearest competitor (any other resonance in the window)
    others = freqs[np.abs(freqs - f_sel) > 1e6]
    sep_hz = float(np.min(np.abs(others - f_sel))) if len(others) else np.inf
    sep_lw = sep_hz / kappa_2pi if np.isfinite(sep_hz) else np.inf

    rows = []
    for f, q, a, e in zip(freqs, Qs, amps, errs):
        is_sel = abs(f - f_sel) <= 1e6
        rows.append({
            "freq_THz": f / 1e12, "detuning_GHz": (f - F_D2) / 1e9,
            "Q": q, "amplitude": a, "error": e, "is_selected_mode": is_sel,
            "separation_linewidths": (0.0 if is_sel else abs(f - f_sel) / kappa_2pi),
        })
    datasets.write_csv(paths.DATA_DIR / "mode_scan.csv", rows,
                       datasets.MODE_SCAN_COLUMNS)

    spec_f, spec_a = _spectrum(sim)
    in_win = (spec_f >= F_D2 - hw) & (spec_f <= F_D2 + hw)
    datasets.write_npz(
        paths.DATA_DIR / "competing_modes.npz",
        spectrum_freq_Hz=spec_f[in_win], spectrum_amp=spec_a[in_win],
        resonance_freq_Hz=freqs, resonance_Q=Qs, resonance_amp=amps,
        f_selected_Hz=np.array([f_sel]),
        kappa_2pi_Hz=np.array([kappa_2pi]),
        separation_Hz=np.array([sep_hz if np.isfinite(sep_hz) else -1.0]),
        separation_linewidths=np.array([sep_lw if np.isfinite(sep_lw) else -1.0]),
    )
    clean = sep_lw >= campaign.MODE_SEPARATION_MIN_LINEWIDTHS
    print(f"[modescan] {len(rows)} resonance(s) in ±{hw/1e12:.1f} THz; "
          f"selected f={f_sel/1e12:.4f} THz")
    print(f"[modescan] nearest competitor: "
          f"{sep_hz/1e9:.1f} GHz = {sep_lw:.1f} linewidths "
          f"({'CLEAN' if clean else 'check'}; need ≥"
          f"{campaign.MODE_SEPARATION_MIN_LINEWIDTHS:.0f})")


if __name__ == "__main__":
    main()
