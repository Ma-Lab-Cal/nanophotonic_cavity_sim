# Publication package manifest

Every figure regenerates from a saved dataset with one command. `./publication_ready/reproduce_all.sh` (no flag) rebuilds all figures and tables from saved data with **zero cloud cost**; `--recompute` re-runs the cloud sweeps/sims.

## Figures

| Figure | Shows | Backing dataset(s) | Regenerate (python -m publication_ready.<cmd>) | Needs --recompute |
|---|---|---|---|---|
| Fig1_device ✓ | Top-view geometry + per-hole a/sₓ/s_y traces (seed vs final) | `data/final_candidate.json`<br>`seeds/seed_780nm.json` | `make_publication_figures --only fig1` | no |
| Fig2_bands ✓ | Bloch bands of mirror & defect cells; gap, light line, D2, f_res | `data/bands_final.json`<br>`data/bands_seed.json` | `make_publication_figures --only fig2` | bands only |
| Fig3_resonance_loss ✓ | Resonance fit (Q, κ) + κ_wg/residual loss budget | `data/nominal_verify.json` | `make_publication_figures --only fig3` | no |
| Fig4_fields ✓ | |E| & Re(E_y) field cuts + field at the atom height | `data/fields_final.npz` | `make_publication_figures --only fig4` | no |
| Fig5_tuning ✓ | Global/defect/output sweeps; overcoupling Pareto | `data/tuning_global_scale.csv`<br>`data/tuning_defect_scale.csv`<br>`data/output_coupling_sweep.csv` | `make_publication_figures --only fig5` | yes |
| Fig6_position ✓ | η & C vs atom height and lateral / along-beam offset | `data/position_map_yz.npz`<br>`data/position_map_xz.npz` | `make_publication_figures --only fig6` | no |
| Fig7_fabrication ✓ | η/β/detuning/Q under fabrication corner perturbations | `data/tolerance_sweep.csv` | `make_publication_figures --only fig7` | yes |
| Fig8_comparison ✓ | Five-design comparison bars (one convention) | `tables/comparison_table.csv` | `make_publication_figures --only fig8` | no |
| SuppA_convergence ✓ | Mesh / PML / run-time convergence | `data/convergence_mesh.csv`<br>`data/convergence_pml.csv`<br>`data/convergence_runtime.csv` | `make_publication_figures --only supp_convergence` | yes |
| SuppB_mode_isolation ✓ | Spectrum + nearest-competitor separation | `data/mode_scan.csv`<br>`data/competing_modes.npz` | `make_publication_figures --only supp_modeiso` | no |

## Source verification simulations (provenance)

| Design | HDF5 | present |
|---|---|---|
| design_1 | `invDesResults/full_seed/verification/baseline_design1.hdf5` | ✓ |
| invDes(design_1) | `invDesResults/full/verification/final_design.hdf5` | ✓ |
| seed | `invDesResults/full_seed/verification/seed_design.hdf5` | ✓ |
| invDes(seed) | `invDesResults/full_seed/verification/final_design.hdf5` | ✓ |
| final (trimmed) | `publication_ready/verification/final_trimmed.hdf5` | ✓ |

## Datasets

| File | Description |
|---|---|
| `tables/comparison_table.csv` | 5-design metrics, one convention |
| `data/nominal_verify.json` | final perf + loss budget + spectrum + settings |
| `data/fields_final.npz` | field cuts on the three profile planes |
| `data/tuning_global_scale.csv` | global-scale resonance trim axis |
| `data/tuning_defect_scale.csv` | defect-only fine-trim axis |
| `data/output_coupling_sweep.csv` | output mirror β–C–η Pareto axis |
| `data/position_map_{yz,xz}.npz` | η/C vs atom height & offset |
| `data/tolerance_sweep.csv` | fabrication corner-grid robustness |
| `data/convergence_{mesh,pml,runtime}.csv` | numerical convergence |
| `data/mode_scan.csv` + `competing_modes.npz` | mode isolation |
| `data/bands_{final,seed}.json` | Bloch unit-cell bands |
| `data/final_design.{json,csv,gds}` | fabrication geometry package |
