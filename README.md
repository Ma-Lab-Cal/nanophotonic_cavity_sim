# nanophotonic_cavity

Design and simulation of **suspended SiN one-dimensional photonic-crystal nanobeam
cavities** resonant with the ⁸⁷Rb D2 line (780 nm / 384.23 THz), intended as a
cavity-QED interface between a tweezer-trapped neutral atom and a waveguide/fiber
photonic channel.

The repository contains three pipelines:

1. **Band-structure simulations** (MPB) — `main_code/bandstructure_mpb.py`, with
   `bandstructure_tutorial.ipynb` as the guide.
2. **Cavity FDTD simulations** (Tidy3D cloud) — the `Cavity` / `Cavity_simulation`
   classes in `main_code/`, with `cavity_tutorial.ipynb` as the guide.
3. **Automated inverse design** (Tidy3D autograd adjoint) — the `inverse_design/`
   package, which optimizes hole spacings, the hole placement schedule, and
   closed-spline hole shapes simultaneously against the physical photon-interface
   figure of merit. See `InverseDesign_Report.pdf` for the design report and
   `Objective.md` (Implementation Log) for the full development history.

## Repository layout

```
main_code/            hand-design pipeline (single source of truth for geometry)
  crystal.py            unit-cell hole polygons (ellipse, rectangle, sawtooth …)
  defect.py             polynomial defect interpolation (mirror -> defect)
  mirror.py, taper.py   uniform / linearly tapered sections
  cavity.py             cavity assembly + gdsfactory GDS export
  simulation.py         Tidy3D FDTD: domain, sources, monitors; Q, directional Q,
                        mode volume, Purcell, g(r), C(r), eta(r) post-processing
  bandstructure_mpb.py  MPB band structures (requires meep/mpb)
inverse_design/       autograd inverse-design package (see below)
tests/                unit tests + cloud-marked adjoint gradient check
0326_TAPEOUT/         1324 nm tapeout designs (Cavity_Tidy wrapper, MPB class)
materials/            measured dispersion data (SiN.txt, Si.txt, SiO2.txt)
cavities_780.ipynb    780 nm design exploration (source of the design_1 baseline)
cavity_tutorial.ipynb / bandstructure_tutorial.ipynb / general_GDS_patterns.ipynb
Objective.md          inverse-design objective + Implementation Log
InverseDesign_Report.pdf  9-page design report (figures, bands, literature context)
```

## The hand-design pipeline

The cavity workflow is organized around a `Cavity` class that generates layouts and
assembles the cavity from component classes (`Taper`, `Mirror`, `Defect`), then a
`Cavity_simulation` class that handles all aspects of the Tidy3D simulations and the
atom-physics post-processing (resonance frequency and Q via `ResonanceFinder`,
per-face directional Q, mode volume, Purcell factor, position-resolved coupling
g(r), cooperativity C(r) = 4g²/(κγ), and collection efficiency
η(r) = (κ_out/κ_tot)·C/(C+1)·η_fiber).

`crystal.py` defines the hole geometry used in the unit cells; to add a new
geometry, extend `crystal_polygon_2d` (and the `is_hole` check in `simulation.py`
if the new geometry is an extrusion rather than a hole). The defect interpolation
scheme lives in `defect.py` (`defect_function`, a cubic polynomial blend by
default). GDS export shares the same polygon vertices as the simulation, via
gdsfactory. `general_GDS_patterns.ipynb` generates alignment markers and other
simple GDS patterns.

## Inverse design (`inverse_design/`)

A fully differentiable design path on top of Tidy3D's native autograd support
(verified on tidy3d 2.11.2), with the existing time-domain pipeline kept as the
non-gradient verification path:

- `parametrization.py` — 278-parameter differentiable geometry: per-cell lattice
  constants, per-hole scales (holes shrinking below ~20 nm count as removed), and
  closed periodic-B-spline hole shapes (radial/star-shaped: closed and never
  self-intersecting), all through bounded smooth transforms.
- `builder.py` — differentiable `td.Simulation` (traced PolySlab vertices;
  frequency-domain monitors only: output-port ModeMonitor, atom-point
  FieldMonitors 200–300 nm above the surface, closed FieldMonitor flux box).
- `objective.py` — loss = −log η̃ + resonance targeting at 384.23 THz +
  fabrication barriers (hole gaps ≥ 50 nm, side rails ≥ 70 nm, curvature radius
  ≥ 40 nm, lattice smoothness).
- `diagnostics.py` — `FrozenGeometryCavitySimulation` injects optimized geometry
  into the unmodified legacy analysis; independent units-checked g(r)/C/η.
- `optimizer.py` / `record.py` — Adam with per-iteration checkpoint/resume, NaN
  and cloud-failure guards, a FlexCredit budget ledger, and a full reconstruction
  record (`params` / `loss` / `time` / `perf` per iteration).
- `bands.py` — TE band structures of unit cells via Tidy3D Bloch simulations
  (used when MPB is unavailable).
- `visualization.py` / `make_report.py` — plots, structure-evolution GIF, GDS
  export, and the PDF report.

### Results

Starting from the reproducible hand-tuned baseline (design_1 in
`cavities_780.ipynb`), 42 optimization iterations (~3.3 h, ~67 FlexCredits)
improved the verified end-to-end collection efficiency at a realistic atom
position (250 nm above the beam) from **η = 0.083 to η = 0.278 (3.4×)** while
moving the resonance from 1.01 THz off the Rb D2 target to 0.29 THz
(C: 21 → 45, β_wg: 0.087 → 0.287, Q: 1137 → 1778). Full analysis, band
structures, field plots, and literature context: `InverseDesign_Report.pdf`.

### Usage

```bash
conda activate tidyEnv                  # tidy3d >= 2.11 with autograd
export TIDY3D_API_KEY=...               # or configure once via tidy3d.web.configure

# cheap local tests (no cloud)
python -m pytest tests/

# ~0.2-credit adjoint-vs-finite-difference gradient check (cloud)
PYTHONPATH=. python tests/test_gradient_fd.py

# end-to-end smoke test (~4 credits)
PYTHONPATH=. python -m inverse_design.run_optimization --mode smoke

# full optimization (resumable; obeys --credit-cap)
PYTHONPATH=. python -m inverse_design.run_optimization --mode full --run-name full
PYTHONPATH=. python -m inverse_design.run_optimization --mode full --run-name full --resume

# unit-cell band diagrams and the PDF report
PYTHONPATH=. python -m inverse_design.bands
PYTHONPATH=. python -m inverse_design.make_report
```

All run outputs (checkpoints, record, verification data, figures, GIF, GDS) land
in `invDesResults/<run-name>/` (gitignored).

## Environment

- Python ≥ 3.12 with `tidy3d ≥ 2.11` (autograd), `autograd`, `numpy`, `scipy`,
  `matplotlib`, `shapely`, `pillow`, `gdsfactory`, `pytest`, `fpdf2` (report).
- MPB/meep only for `bandstructure_mpb.py` (optional; `inverse_design/bands.py`
  is the Tidy3D-based alternative).
- A Tidy3D cloud API key is required for FDTD runs.
