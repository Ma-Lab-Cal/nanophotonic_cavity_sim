"""Publication-readiness campaign for the SiN nanobeam Rb cavity.

This package takes the inverse-designed ``invDes(seed)`` cavity
(``invDesResults/full_seed/``) to a publishable photonic-design claim:

* a resonance trim onto the ⁸⁷Rb D2 line (384.23 THz),
* a complete EM-only dataset + figure package,
* a one-command, data-only figure-regeneration harness.

It **imports** the existing ``inverse_design`` and ``main_code`` APIs and never
modifies them. The only new physics-adjacent code lives in
``publication_ready.lib`` (layout scaling, field postprocessing, plotters,
dataset writers). All metrics use the single ``inverse_design.diagnostics``
convention; see ``publication_ready/limitations.md``.
"""
