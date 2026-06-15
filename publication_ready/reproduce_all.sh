#!/usr/bin/env bash
# Regenerate the publication package.
#
#   ./reproduce_all.sh                 # data-only: rebuild every figure/table
#                                      # from saved datasets (NO cloud cost)
#   ./reproduce_all.sh --recompute     # also re-run the cloud sweeps/sims
#   TIER=core ./reproduce_all.sh --recompute
#
# The cheap stage fails loudly if a required dataset/HDF5 is missing rather than
# silently launching cloud simulations.
set -euo pipefail
cd "$(dirname "$0")/.."                 # repo root
export PYTHONPATH=.
PY="${PYTHON:-/home/ma-lab/anaconda3/envs/tidyEnv/bin/python}"
RECOMPUTE="${1:-}"

run() { echo "+ $*"; "$PY" -m "$@"; }

if [ "$RECOMPUTE" = "--recompute" ]; then
  echo "== EXPENSIVE STAGE: cloud simulations =="
  run publication_ready.run_tuning_sweeps          # global + defect trim -> final_candidate.json
  run publication_ready.run_output_coupling        # Fig-5 output axis (+ rebalance if gate failed)
  run publication_ready.nominal_verify             # 1 high-res final sim + fields + geometry pkg
  run publication_ready.run_bands                  # final cells (seed cells reused)
  run publication_ready.run_convergence            # mesh / PML / run-time
  run publication_ready.run_tolerance              # fabrication corner grid
fi

echo "== CHEAP STAGE: postprocess saved data (no cloud cost) =="
run publication_ready.build_comparison_table       # reuses 4 saved HDF5
run publication_ready.run_position_maps            # postprocess final sim
run publication_ready.run_mode_scan                # postprocess final sim

echo "== FIGURES + REPORT + MANIFEST (pure data -> pdf) =="
run publication_ready.make_publication_figures
run publication_ready.gen_manuscript_notes
run publication_ready.publication_report
run publication_ready.build_manifest

echo "Done. See publication_ready/figures/, tables/, Publication_Report.pdf, "
echo "and publication_ready_manifest.md"
