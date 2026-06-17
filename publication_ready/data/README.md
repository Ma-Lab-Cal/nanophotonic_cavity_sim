# `publication_ready/data/` — final design and validated results

This directory holds the **final device geometry** and the **validated electromagnetic
figures of merit** for the inverse-designed suspended-SiN photonic-crystal nanobeam
cavity resonant with the ⁸⁷Rb D2 line (780.241 nm / 384.231 THz). Everything here is
derived from classical Tidy3D (FDTD) simulations of the final geometry — there is no
atomic-dynamics / cavity-QED simulation (see *Conventions & caveats* below).

---

## ⭐ The final device

| File | What it is |
|---|---|
| **`final_design.gds`** | **THE deliverable.** Complete nanobeam: 500 nm wide × 150 nm thick, 80 holes (5 left taper + 4 left mirror + 40 graded defect + 30 right mirror + 1 right taper), ~25.4 µm hole array (~29 µm beam). The conformal in-plane trim (`global_scale = 1.00124`) is **already applied**, so the simulated resonance sits on the D2 line. A `$$$CONTEXT_INFO$$$` cell carries metadata. |
| `final_candidate.json` | Authoritative record of the final layout: hole positions, lattice, hole sizes (`layout`), the applied `global_scale`, the cell counts, and the pass/fail `gate` (frequency, β, η, C, single-mode all True; residual gate detuning 6.4 MHz). |
| `final_design.json` | Human-readable summary of the same device (cell counts, width/thickness, per-hole list, target frequency, and a `perf` block — **see dipole caveat**). |
| `final_design.csv` | Per-hole geometry table: `hole_index, section, x_center_um, lattice_a_nm, sx_nm, sy_nm`. Pure geometry; unaffected by any physics convention. |

> The `final_design.gds` files under `invDesResults/full_seed|full|smoke/` are **not** the
> deliverable — `full_seed/final_design.gds` is the raw optimizer output *before* the trim
> (~0.12% smaller → ~+300 GHz off D2); the others are earlier/intermediate runs.

---

## Headline figures of merit (corrected convention)

| Quantity | Value | Notes |
|---|---|---|
| Resonance detuning \|f_res − f_D2\| | **0.3 GHz** (±23 GHz numerical) | within one cavity linewidth |
| Loaded Q / κ/2π | 1.2×10⁴ / **31–32 GHz** | ring-down |
| Branching ratio β | **0.88** | directional-flux into the −x output waveguide |
| Cooperativity C | **≈ 12** | C = 4g²/κγ |
| Coupling g/2π | **≈ 0.77 GHz** | dipole-aligned upper bound |
| Coupling g/2π (σ⁺ cycling) | ≈ 0.55 GHz | linear-mode σ⁺ value (C ≈ 6) |
| Chip efficiency η_chip = β·C/(C+1) | **0.82** | — |
| With-fiber efficiency η | **0.81** | assumes η_fib = 0.99 |
| Mode volume V | ≈ 0.87 (λ/n)³ | — |
| Atom height | 250 nm above top surface | fixed |
| Fab-corner robustness | η = 0.78–0.84, β = 0.86–0.89 | etch ±4 nm, lattice ±2 nm, width ±10 nm |

These match the manuscript. An independent high-accuracy validation run (`validation/`)
gives g ≈ 0.72 GHz, C ≈ 11, η_chip ≈ 0.81 — consistent within the stated ~10 % mesh
sensitivity of g/C.

---

## ✅ Validated outputs (authoritative, post–dipole-correction)

The files in **`validation/`** were regenerated with the corrected code and are the
authoritative validated results:

| File | What it validates |
|---|---|
| **`validation/validation_table.md` / `.csv` / `.json`** | The full auditable parameter table (corrected dipole convention), with units and definitions for every FOM. **Start here.** |
| `validation/grid_offset.json` | The resonance claim: grid-offset-averaged, calibrated re-trim → \|f_res − f_D2\| = 0.3 GHz, `within_one_linewidth = true`. |
| `validation/freq_resolution.json` / `_ladder.csv` | Mesh-convergence ladder of f_res (the single-mesh scatter → converged fine-mesh mean). |
| `validation/convergence_cross_checks.json` | Run-time / PML / domain / symmetry cross-checks. |
| `validation/single_mode_spectrum.json` | Single-mode confirmation (nearest competing resonance > 1 THz away). |
| `validation/fields_x|y|z.pdf` / `.png` | High-resolution field maps of the final mode. |
| `validation/*.log` | Console logs of the billable validation runs (audit trail). |

Regenerate with `python validate_design.py --all` (billable Tidy3D runs).

---

## Supporting design-study data (raw, pre–dipole-correction for g/C/η)

These document the design study and feed the manuscript figures. **The g/C/η columns in
these files use the J-reduced dipole (the pre-correction convention)** — see the caveat.

- `nominal_verify.json` — production-pipeline `perf` of the final device.
- `comparison_table.json` — design lineage (baseline → inverse design → seed → final).
- `bands_final.json`, `bands_seed.json` — Bloch band structures (mirror & defect cells).
- `output_coupling_sweep.{json,csv}` — the over-coupling trade-off (β vs C vs η).
- `tuning_global_scale.{json,csv}` — the resonance-trim knob (detuning vs in-plane scale).
- `tuning_defect_scale.{json,csv}` — defect-only scale sweep.
- `tolerance_sweep.csv`, `tolerance_samples/*.json` — fabrication corner grid.
- `convergence_{mesh,pml,runtime}.csv` — convergence studies.
- `fields_final.npz` — saved field arrays (|E|, Ey) used for the manuscript field figure.
- `position_map_{xz,yz}.npz` — g/C/η vs atom position maps.
- `mode_scan.csv`, `competing_modes.npz` — spectral single-mode check.

---

## ⚠️ Conventions & caveats (read before using any number)

1. **Dipole convention (affects g, C, η — NOT β, Q, κ, V, f_res).**
   The headline coupling uses the ⁸⁷Rb D2 **σ⁺ cycling dipole**
   `d = √(1/2)·⟨J‖er‖J'⟩ = 2.53×10⁻²⁹ C·m`. The raw per-design files above
   (`final_design.json`, `nominal_verify.json`, `comparison_table.json`,
   `output_coupling_sweep.*`, `tuning_*`, `tolerance_*`, `position_map_*`) store g/C/η
   computed with the **J-reduced element** `3.58×10⁻²⁹ C·m` — i.e. **g is a factor √2 too
   large and C a factor 2 too large** in those files (g_2pi_Hz_at_250nm = 1.09×10⁹,
   C_atom = 24.4). To get the reported values, rescale exactly:
   `g → g·√(1/2)`, `C → C·(1/2)`, `η = β·C/(C+1)·η_fib`.
   The `validation/validation_table.*` files already carry the corrected values.

2. **EM proxies, not cavity-QED.** g, C, β, η are fixed-position, two-level,
   dipole-aligned scalar proxies from classical fields — they bound the interface, not
   gate/readout/entanglement fidelities.

3. **g is a dipole-aligned upper bound.** It assumes the atomic dipole is aligned with the
   local (linearly polarized) cavity field; a specific σ⁺ cycling transition couples a
   further 1/√2 weaker (the σ⁺ value is tabulated).

4. **η assumes η_fib = 0.99** (waveguide-to-fiber) and the fixed 250 nm trap height.

5. **Absolute resonance** carries the ~23 GHz numerical uncertainty; the in-plane scale
   (`global_scale`, panel g of the performance figure) is a pre-fabrication knob to center
   the as-fabricated resonance (f_res ∝ 1/s).

Each headline JSON embeds a `convention` block (from `publication_ready/lib/datasets.py`)
recording the exact metric definitions, target frequency, and assumptions.
