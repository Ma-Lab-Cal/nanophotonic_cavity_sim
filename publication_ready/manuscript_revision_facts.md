# Manuscript revision facts (verified)

Verified facts for the Section-5 manuscript revision, from an adversarially-checked
literature/physics workflow (gather → skeptic). Each item was confirmed against the local
PDFs or computed analytically. Use these when revising `paper/main.tex`.

## 1. Clebsch–Gordan / dipole factors (87Rb D2)

- Reduced dipole `mu = 3.58e-29 C·m` is the **J reduced matrix element** ⟨J=1/2‖er‖J'=3/2⟩
  = 4.22776 e a₀ (Steck, *87Rb D Line Data*). It is **NOT** the effective cycling dipole
  (~2.54e-29 C·m). `config.py:23` label fixed; `validate_design.py` correct.
- Cycling-normalized **amplitude** factors (linear multiplier on g), corrected:

  | Transition | Pol | factor |
  |---|---|---|
  | \|2,2⟩→\|3,3⟩ cycling | σ+ | 1.000 |
  | \|2,0⟩→\|3,0⟩ | π | 0.775 = √(3/5) |
  | \|2,0⟩→\|2,0⟩ | π | 0.000 (dipole-forbidden) |
  | \|2,0⟩→\|1,0⟩ | π | 0.394 = √(7/45) |
  | isotropic / pol-averaged | — | 0.577 = 1/√3 |

  (Manuscript/old-code values 0.745/0.456/0.373 were wrong; the gather-agent's own
  proposed 0.258 for the 2→1 line was also wrong. `validate_design.py` table fixed.)
- Methods sentence: "g = g_reduced · (transition factor) · |ê·Ê|, with g_reduced from the
  J reduced dipole; the reported g uses the σ+ cycling transition (factor 1), so g scales
  DOWN for other transitions/polarizations and is not a universal Rb D2 coupling."

## 2. Related-work benchmark numbers (for intro/results; all verified vs local PDFs)

**CRITICAL FRAMING FIX.** Thompson et al. 2013 (Science **340**, 1202–1205) report **"0.07"
as the COOPERATIVITY** (their η ≡ (2g)²/(κΓ) = 4g²/κΓ), *not* a collection efficiency. So the
old manuscript headline "η: 0.07 → 0.84" conflates a cooperativity with an efficiency.
- Both Thompson AND the local review define C = 4g²/(κγ) — **same convention as ours** — so a
  *cooperativity* comparison is valid: **C ≈ 0.07 (Thompson) → C ≈ 24 (this work)**.
- Branching/cavity-emission efficiency: C/(C+1) = 0.065 (Thompson) → 0.96 (this work). Our
  η_chip = β·C/(C+1) ≈ 0.85 folds in the waveguide branching β; η_with_fiber adds η_fib.
- **Recommended manuscript wording:** compare cooperativity (C 0.07→24, ~340×, both 4g²/κγ),
  then state η as an absolute EM-proxy. Do NOT call Thompson's 0.07 an "efficiency."
- González-Tudela 2024 uses C = g²/(κγ) (factor 4 smaller) — note the convention when citing.

Benchmark FOMs (verified):
- **Thompson 2013** (single-atom SiN PhC cavity): (2g,κ,Γ)/2π = [0.60(8), 840(80), 0.006] GHz;
  C = 0.07(1); Q = 460(40) at λ=779.5 nm (2nd-order mode); V = 0.89 λ³ (used) / 0.42 λ³
  (fundamental); measured fiber collection (1.5±0.6)%; η_wg = 0.06. Projected (not achieved):
  2g/2π→3 GHz, SiN Q up to 3×10⁵, C > 1000.
- **Tiecke 2014** (Harvard PhC, via Reiserer&Rempe RMP): C ≃ 4, κ/2π ≃ 13 GHz, κ = 25g.
- **González-Tudela 2024** review: C ≈ 0.05–0.1 (nanofibres), 1–2 (PhC waveguides), O(10)
  near-term; β = Γ₁D/(Γ₁D+γ) = C/(C+1).
- **Đorđević 2021** (Lukin/Vuletić PhC, in local review Table 1): g/2π=620 MHz, κ_tot/2π=3630
  MHz, γ/2π=6 MHz, **C = 71 demonstrated**.
- **Li & Thompson 2024** (PRX Quantum, bow-tie *proposal*): g/2π=0.52 MHz, κ_tot/2π=1.04 MHz,
  γ/2π=0.418 MHz, C=2.5.
- **Menon 2020** (NJP 22, 073033 = arXiv:2002.05175, *theory*): FDTD C 5–260, avg C=35 at
  Q=2×10⁵; fidelity >0.9 for C≳15. Definition C = 2g²/(κ(γ₂+γ₃)).
- **Menon 2024** (Nat Commun 15, 6156): imaging platform, cavities NOT yet resonant → no
  measured C/Q; projected g ~ 2π×600 MHz at ~300 nm; ~60 µm cavities.
- **Covey 2019** PRApplied (Yb-on-Si): NOT in local folder; cite from bibliography only.

## 3. Over-coupling sentence (FIX 1) — physics correct, paste-ready

Replace the reversed Results sentence (`main.tex:~408–410`):

> As the output mirror is made more transparent and the external coupling $\kappa_\mathrm{wg}$
> is increased, the loaded $Q=\omega_c/\kappa_\mathrm{tot}$ and the cooperativity
> $C=4g^2/(\kappa\gamma)$ decrease monotonically, while the branching ratio
> $\beta=\kappa_\mathrm{wg}/\kappa_\mathrm{tot}$ rises and approaches (but never reaches) unity
> as $\kappa_\mathrm{wg}$ comes to dominate the residual loss. Because
> $\eta=\beta\,[C/(C{+}1)]\,\eta_\mathrm{fib}$ multiplies a rising $\beta$ against a falling
> cavity-emission factor $C/(C{+}1)$, the efficiency optimum sits at intermediate,
> strong-but-not-maximal coupling, coinciding with neither maximal $Q$ nor maximal $C$.

(Call C/(C+1) the *cavity-emission/Purcell efficiency*, NOT "the Purcell factor" — that symbol
is already F_P = 3/(4π²)Q/V in the manuscript. Make Fig.(h) sweep direction explicit.)

## 4. β definition (FIX 2) — corrected to match the code

Keep the **headline β = 0.886 as β_flux** (directional). Present β_guided/β_fund as cross-checks.
Replace `main.tex:~196–201`:

> the branching ratio $\beta=\kappa_\mathrm{wg}/\kappa_\mathrm{tot}$, the cavity decay into the
> output ($-x$) waveguide. The headline value is the directional estimator
> $\beta_\mathrm{flux}=\kappa(-x)/\sum_\mathrm{faces}\kappa_\mathrm{dir}$. As stricter
> cross-checks a mode-expansion monitor (ModeMonitor) co-located with a flux monitor in the
> output waveguide projects the collection-plane field onto the precomputed guided modes,
> giving $\beta_\mathrm{guided}=\beta_\mathrm{flux}(P_\mathrm{guided}/P_\mathrm{plane})$ with
> $P_\mathrm{guided}=\sum_m|a_m|^2$, and the fundamental-mode value
> $\beta_\mathrm{fund}=\beta_\mathrm{flux}(|a_0|^2/P_\mathrm{plane})$. Since $\beta_\mathrm{flux}$
> also counts radiation crossing the plane,
> $\beta_\mathrm{flux}\ge\beta_\mathrm{guided}\ge\beta_\mathrm{fund}$; the higher-order guided
> content $1-|a_0|^2/\sum_m|a_m|^2$ is reported separately.

Caveats (verified):
- β_guided in the code sums over ALL guided modes (not fundamental-only); fundamental is
  β_fund. Code fixed to report both.
- The directional sum Σκ_dir ≈ 154 GHz does NOT equal the ring-down linewidth κ_tot=ω/Q ≈ 32
  GHz; β is a ratio so the normalization cancels, but do **not** claim Σκ_dir = κ_linewidth.

## 5. Operational-impact softening (change-request Results)

- Drop "more than two orders of magnitude entanglement-rate gain" as a prediction; if kept,
  label explicitly as a face-value η² projection from EM proxies.
- Relabel "~ln C/C ≈ 13% infidelity" as a heuristic scaling, not evaluated here.
- Recompute any comparison using the *cooperativity* framing of §2 above.
