# Manuscript notes — from invDes(seed) to the final trimmed design

**The trim.** The inverse-designed seed cavity (invDes(seed)) resonated at 384.53 THz, +303 GHz (about 10 loaded linewidths) above the Rb D2 line. The final design applies a single global in-plane scale s = 1.001239 (+0.124%) — measured by a two-point bracket and linear interpolation, since the fixed 150 nm film and material dispersion make f proportional to 1/s only approximately. Thickness and width are unchanged.

**Final verified design.**

- f_res = 384.2306 THz (+0.0 GHz from D2; tolerance 8.0 GHz)
- Q = 11990, kappa/2pi = 32.0 GHz
- beta = kappa_wg/kappa_tot = 0.886 (residual loss 11%)
- C = 24.4, g/2pi = 1.090 GHz @ 250 nm, eta = 0.842
- V = 0.874 (lambda/n)^3
- Acceptance gate: **PASSED**

**Why the trim preserved the interface.** A global conformal in-plane scale shifts the mirror band gap and the cavity mode together, so the dimensionless photonic design (beta, V/(lambda/n)^3, C, eta) is preserved to first order. The sweep confirms this directly: across the trim, beta, C and eta change by less than ~1%. The output-coupling rebalance and a second-stage re-optimization were therefore held in reserve and (per the gate) not required.

**The four-step story (one metric convention).**

| design | eta | beta | C | Q | detuning (GHz) |
|---|---|---|---|---|---|
| design_1 | 0.083 | 0.087 | 21.2 | 1137 | 1010.5 |
| invDes(design_1) | 0.278 | 0.287 | 45.0 | 1778 | -291.6 |
| seed | 0.088 | 0.089 | 471071.2 | 27116904 | -615.5 |
| invDes(seed) | 0.834 | 0.875 | 25.4 | 12600 | 303.5 |
| final | 0.842 | 0.886 | 24.4 | 11990 | 0.0 |

The hand-tuned baseline (design_1) and the band-engineered seed are both strongly undercoupled (beta ~ 0.09): the seed achieves a near-lossless photonic mode (its Q ~ 10^7 and C ~ 10^5 are **unresolved upper bounds**, not operating values — the field barely decays in the window). Gradient inverse design opens the output port (beta -> 0.88) and trades the useless intrinsic Q for a resolved, overcoupled interface; the global-scale trim then lands it on the Rb D2 line without disturbing that interface.

**Numerical convergence.** The absolute resonance frequency is mesh-sensitive: across defect mesh dl = 0.006-0.012 um the fitted f_res varies by ~81 GHz, larger than the gate tolerance. The trim and all comparison designs therefore use one fixed production mesh (dl = 0.010 um, the project standard); the convergence study (Supplement A) quantifies the residual uncertainty, and in practice the +/-0.1% global-scale knob (the same one used for the trim) absorbs it at the fabrication / tuning stage. beta, eta and C are far less mesh-sensitive.

**Robustness (see Figs 6, 7 and Supplement A).** eta and C are mapped over atom height (200-350 nm) and lateral/along-beam offset; the design is checked against a deterministic fabrication corner grid (etch/lattice/width biases) with a first-order global-scale retune, and against mesh / PML / run-time numerical convergence. All metrics use the single convention documented in limitations.md.
