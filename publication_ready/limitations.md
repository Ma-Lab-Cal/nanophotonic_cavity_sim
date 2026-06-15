# Scope, conventions, and limitations

This package supports a **photonic-design claim**: a manufacturable suspended-SiN
1D photonic-crystal nanobeam cavity, resonant with the ⁸⁷Rb D2 line, with high
waveguide extraction and robust EM-derived interface figures of merit. Every
number is computed from classical electromagnetic simulation (Tidy3D FDTD) and
scalar postprocessing. The conventions below are used uniformly across **all**
figures, datasets, and the five-design comparison.

## Scope (what this is, and is not)

* **In scope (EM-only):** wave/eigenmode/time-domain/frequency-domain
  verification, Bloch band structure, port-flux/mode-volume postprocessing,
  geometry sweeps, and the scalar atom–light proxies below.
* **Out of scope:** full cavity-QED / master-equation dynamics, atom motion,
  Raman/STIRAP, hyperfine control, and **any gate, readout, or entanglement
  fidelity**. The atom–light metrics (g, C, η) are **EM-derived scalar proxies**
  from the classical fields, not outputs of a quantum-dynamics simulation. The
  manuscript language must not claim qubit/network performance.

## Single metric convention (`inverse_design.diagnostics.perf_from_simulation`)

* **Resonance & Q.** f_res and Q from a ResonanceFinder (Harminv-style) fit of
  the time-domain ring-down over a ±8 THz window; linewidth κ_tot = 2π·f_res/Q.
* **β (the central interface metric).** β = κ(−x)/κ_directional(total), where the
  directional rates κ(face) = ω_c·E_stored/Φ(face) come from the six planar flux
  monitors. **The output port is −x** (the few-cell, left mirror); β is the
  fraction of cavity loss exiting the intended waveguide. Both estimators of the
  total rate are reported: the ring-down linewidth κ_tot and the directional
  flux total; they need not coincide.
* **Mode volume.** V = ∫ε|E|² dV / max(ε|E|²), in (λ/n)³. **No legacy symmetry
  factor** is applied (the asymmetric one-sided cavity does not warrant the
  2^(#sym) multiplier that `main_code.simulation.mode_volume` includes for
  convention continuity); quoting the legacy convention would inflate V by ~2×.
* **g (coupling).** g(r) = d·√(ω·u_pol(r) / (2 ħ ε₀ ∫ε|E|² dV)), with the Rb D2
  effective dipole d = 3.58×10⁻²⁹ C·m, u_pol = |E_y|² time-averaged at the atom
  (vacuum, ε_r=1), sampled on the high-resolution x=0 yz profile monitor.
  Reported as **g/2π in Hz**. Any factor-of-2 dipole convention is fixed by this
  definition.
* **C and η.** C = 4g²/(κ_tot·γ) with the Rb D2 natural linewidth
  γ = 2π·6.0666 MHz; η = β·C/(C+1)·η_fiber with η_fiber = 0.99 (collection
  efficiency). The detuned correction η_eff = η/[1+(2Δ/κ)²] is shown only as an
  interpretive Lorentzian suppression for off-resonant candidates.

All five designs (design_1, invDes(design_1), seed, invDes(seed), final trimmed)
are recomputed through this single path from their saved simulation files, so
the comparison is apples-to-apples.

## Method limitations and caveats

* **In-plane-only resonance trim.** The film thickness (150 nm) and beam width
  (500 nm) are fixed fabrication parameters; the trim scales only in-plane
  dimensions. Frequency therefore scales as f ≈ 1/s only **approximately**
  (fixed thickness + dispersive SiN break exact scale invariance), so the trim
  is an empirically measured bracket-and-interpolate, not a single analytic
  scale. A full-3D-scaled variant (which would scale the film too) is **not**
  fabrication-realizable post-deposition and is not used.
* **Atom-position maps hold κ and β from the cavity.** η and C across the
  (height, offset) grid are re-derived only from the **local** field; κ and β are
  not re-solved per position. This is the intended scalar postprocessing, not a
  per-position eigenmode solve.
* **Fabrication retuning is first-order, not re-optimization.** For each
  perturbed geometry the "retuned" detuning uses the analytic global-scale
  factor f_res/f_target (which restores D2 to first order and preserves β/η,
  since a conformal scale preserves the dimensionless design). Both the
  unretuned and retuned values are reported; the retune is not a fresh gradient
  optimization.
* **High-Q numbers are unresolved upper bounds.** The band-engineered **seed**'s
  Q (~10⁶–10⁷) and cooperativity (~10⁵) are **not** physical operating values:
  at that Q the field barely decays within the simulation window, so the fit is
  an unresolved floor. The seed is included only to show the photonic starting
  point; the **operating** design is the overcoupled, resolved-Q trimmed cavity.
* **Convergence coverage.** The Core convergence study spans mesh density, PML
  layer count, and run-time/fit-window — the axes that dominate the f_res/Q/η
  numerical uncertainty for this resolved-Q cavity. Domain padding is held at the
  verified 2λ of the production pipeline.
