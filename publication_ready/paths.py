"""Canonical output locations and the saved-HDF5 design registry.

Single source of truth for *where* every dataset/figure goes and *which*
existing simulation file backs each of the four prior designs. Keeping this in
one module means the sweep drivers, figure builders, and the reproduction
harness all agree on layout without hard-coding paths.
"""

from __future__ import annotations

from pathlib import Path

from inverse_design import config as cfg

# ── Roots ──────────────────────────────────────────────────────────────────
REPO_ROOT = cfg.REPO_ROOT
PUB_ROOT = REPO_ROOT / "publication_ready"
DATA_DIR = PUB_ROOT / "data"
FIG_MAIN = PUB_ROOT / "figures" / "main"
FIG_SUPP = PUB_ROOT / "figures" / "supplement"
TABLES_DIR = PUB_ROOT / "tables"
# local cache for cloud verification sims produced by the campaign (gitignored)
VERIFY_DIR = PUB_ROOT / "verification"
TOL_SAMPLES_DIR = DATA_DIR / "tolerance_samples"


def ensure_dirs():
    for d in (DATA_DIR, FIG_MAIN, FIG_SUPP, TABLES_DIR, VERIFY_DIR,
              TOL_SAMPLES_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ── Registry of the four prior designs (existing saved HDF5) ───────────────
# Each entry: label -> (hdf5 path, record dir or None, source note).
# The 5th design ("final") is produced by the resonance trim and registered at
# runtime via final_candidate.json.
RESULTS = cfg.RESULTS_ROOT

HDF5_REGISTRY = {
    "design_1": RESULTS / "full_seed" / "verification" / "baseline_design1.hdf5",
    "invDes(design_1)": RESULTS / "full" / "verification" / "final_design.hdf5",
    "seed": RESULTS / "full_seed" / "verification" / "seed_design.hdf5",
    "invDes(seed)": RESULTS / "full_seed" / "verification" / "final_design.hdf5",
}

# Record directories (for geometry reconstruction / trajectories).
RECORD_DIRS = {
    "invDes(design_1)": RESULTS / "full",
    "invDes(seed)": RESULTS / "full_seed",
}

# The design we trim onto resonance.
FINAL_SOURCE_LABEL = "invDes(seed)"
FINAL_RECORD_DIR = RESULTS / "full_seed"

# Pre-existing band data we can reuse without re-simulating.
SEED_BANDS_JSON = RESULTS / "seed_design" / "bands.json"

# Final-design artifacts produced by the trim.
FINAL_CANDIDATE_JSON = DATA_DIR / "final_candidate.json"
FINAL_TRIMMED_HDF5 = VERIFY_DIR / "final_trimmed.hdf5"   # without .hdf5 dup; see note
FINAL_TRIMMED_NAME = "final_trimmed"                      # save_name (dir = VERIFY_DIR)


def final_hdf5_path() -> Path:
    """Where the trimmed final-design verification sim is saved."""
    return VERIFY_DIR / f"{FINAL_TRIMMED_NAME}.hdf5"
