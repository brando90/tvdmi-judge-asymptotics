/-
  TVD-MI Judge Ensembles — Signal-Deconfounded Shrinkage (`SRB-Spec`), Partial Proxy Covariates,
  Multi-Factor `Rank-r` Spectral Perturbation, & Continuous Measure-Theoretic TVD (11-21-2026)

  Machine-verifies Experiment 15 mathematical theorems (`12` theorems, zero `sorry`):
    Section A: `SRB-Spec` Signal-Deconfounding & Interior Convex Shrinkage Dominance
    Section B: Imperfect Proxy Covariate `(1 - r^2) * c^2 * v_W` Residual Floor Law
    Section C: Multi-Factor `Rank-r` Spectral Subspace Perturbation & `O(n^{-1/2})` Rate
    Section D: Continuous Measure-Theoretic TVD Event Witness & Partition Refinement Monotonicity
-/

import Mathlib.Tactic
import Mathlib.Data.Real.Sqrt
import Mathlib.MeasureTheory.Measure.MeasureSpace
import Mathlib.MeasureTheory.Integral.Bochner.Set
import TvdmiJudge.SpectralCalibration

open scoped BigOperators
open Filter MeasureTheory

/-!
## Section A: `SRB-Spec` Signal-Deconfounding & Interior Shrinkage Dominance
-/

section SignalDeconfoundingShrinkage

/-- Marginal judge covariance mixes latent quality alignment `a j * a k` with shared family
    cloning `τ_sq (fam j)`: across distinct families (`fam j ≠ fam k`), marginal covariance
    is strictly positive whenever `0 < a j` and `0 < a k`, whereas residualizing out the
    signal outer product `a j * a k` zeroes cross-family correlation while isolating `τ_sq`. -/
theorem marginal_cov_confounds_signal_and_cloning {K F : ℕ} (fam : Fin K → Fin F)
    (a : Fin K → ℝ) (τ_sq : Fin F → ℝ) (j k : Fin K)
    (hdiff : fam j ≠ fam k) (haj : 0 < a j) (hak : 0 < a k) :
    let cov := fun u v : Fin K => a u * a v + if fam u = fam v then τ_sq (fam u) else 0
    0 < cov j k ∧ cov j k - a j * a k = 0 ∧ cov j j - a j * a j = τ_sq (fam j) := by
  dsimp only
  refine ⟨?_, ?_, ?_⟩
  · simp [hdiff, mul_pos haj hak]
  · simp [hdiff]
  · simp

/-- Closed-form quadratic MSE risk expansion for convex shrinkage `(1 - λ) * eU + λ * eB`. -/
theorem convex_shrinkage_mse_formula {n : ℕ} (hn : 0 < n) (eU eB : Fin n → ℝ) (lam : ℝ) :
    (∑ i, ((1 - lam) * eU i + lam * eB i) ^ 2) / (n : ℝ) =
      (1 - lam) ^ 2 * ((∑ i, (eU i) ^ 2) / (n : ℝ)) +
        2 * (1 - lam) * lam * ((∑ i, eU i * eB i) / (n : ℝ)) +
        lam ^ 2 * ((∑ i, (eB i) ^ 2) / (n : ℝ)) := by
  have hnpos : (n : ℝ) ≠ 0 := (Nat.cast_pos.mpr hn).ne'
  have hpt : ∀ i : Fin n,
      ((1 - lam) * eU i + lam * eB i) ^ 2 =
        (1 - lam) ^ 2 * (eU i) ^ 2 + 2 * (1 - lam) * lam * (eU i * eB i) +
          lam ^ 2 * (eB i) ^ 2 := by
    intro i; ring
  simp_rw [hpt, Finset.sum_add_distrib, ← Finset.mul_sum]
  field_simp [hnpos]

/-- Whenever uniform and balanced estimators have covariance strictly below both marginal
    risks (`cUB < vU` and `cUB < vB`), the interior optimal shrinkage weight
    `λ⋆ = (vU - cUB) / (vU + vB - 2 * cUB)` satisfies `0 < λ⋆ < 1` and strictly improves on
    both endpoints (`risk λ⋆ < vU` and `risk λ⋆ < vB`). -/
theorem interior_shrinkage_strictly_dominates_both (vU vB cUB : ℝ)
    (hU : cUB < vU) (hB : cUB < vB) :
    let den := vU + vB - 2 * cUB
    let lam_star := (vU - cUB) / den
    let risk := fun lam : ℝ => (1 - lam) ^ 2 * vU + 2 * (1 - lam) * lam * cUB + lam ^ 2 * vB
    0 < lam_star ∧ lam_star < 1 ∧ risk lam_star < risk 0 ∧ risk lam_star < risk 1 := by
  dsimp only
  set den := vU + vB - 2 * cUB
  set lam_star := (vU - cUB) / den
  set risk := fun lam : ℝ => (1 - lam) ^ 2 * vU + 2 * (1 - lam) * lam * cUB + lam ^ 2 * vB
  set a := vU - cUB
  set b := vB - cUB
  have ha : 0 < a := sub_pos.mpr hU
  have hb : 0 < b := sub_pos.mpr hB
  have hs : 0 < a + b := add_pos ha hb
  have hs_ne : a + b ≠ 0 := hs.ne'
  have hden_eq : vU + vB - 2 * cUB = a + b := by dsimp [a, b]; ring
  have h1m : 1 - lam_star = b / (a + b) := by
    dsimp [lam_star, den]
    rw [hden_eq, one_sub_div hs_ne]
    ring
  have hlam_eq : lam_star = a / (a + b) := by dsimp [lam_star, den]; rw [hden_eq]
  have hvU : vU = a + cUB := by dsimp [a]; ring
  have hvB : vB = b + cUB := by dsimp [b]; ring
  have hcancel : (a + b) * (a + b)⁻¹ = 1 := mul_inv_cancel₀ hs_ne
  have hpoly0 : b ^ 2 * vU + 2 * a * b * cUB + a ^ 2 * vB =
      (a + b) ^ 2 * vU - a ^ 2 * (a + b) := by
    rw [hvU, hvB]; ring
  have hpoly1 : b ^ 2 * vU + 2 * a * b * cUB + a ^ 2 * vB =
      (a + b) ^ 2 * vB - b ^ 2 * (a + b) := by
    rw [hvU, hvB]; ring
  have hrisk_fact : risk lam_star =
      (b ^ 2 * vU + 2 * a * b * cUB + a ^ 2 * vB) * (a + b)⁻¹ ^ 2 := by
    dsimp [risk]
    rw [h1m, hlam_eq]
    simp only [div_eq_mul_inv]
    ring
  have hrisk0 : risk lam_star = vU - a ^ 2 * (a + b)⁻¹ := by
    rw [hrisk_fact, hpoly0]
    have hpow2 : ((a + b) ^ 2 * vU - a ^ 2 * (a + b)) * (a + b)⁻¹ ^ 2 =
        vU * ((a + b) * (a + b)⁻¹) ^ 2 - a ^ 2 * (a + b)⁻¹ * ((a + b) * (a + b)⁻¹) := by
      ring
    rw [hpow2, hcancel]
    ring
  have hrisk1 : risk lam_star = vB - b ^ 2 * (a + b)⁻¹ := by
    rw [hrisk_fact, hpoly1]
    have hpow2 : ((a + b) ^ 2 * vB - b ^ 2 * (a + b)) * (a + b)⁻¹ ^ 2 =
        vB * ((a + b) * (a + b)⁻¹) ^ 2 - b ^ 2 * (a + b)⁻¹ * ((a + b) * (a + b)⁻¹) := by
      ring
    rw [hpow2, hcancel]
    ring
  have hr0_eval : risk 0 = vU := by dsimp [risk]; ring
  have hr1_eval : risk 1 = vB := by dsimp [risk]; ring
  have hdiv1 : lam_star < 1 := by
    rw [hlam_eq, div_lt_one hs]
    linarith
  have hdiv0 : 0 < lam_star := by
    rw [hlam_eq]
    exact div_pos ha hs
  refine ⟨hdiv0, hdiv1, ?_, ?_⟩
  · have hpos : 0 < a ^ 2 * (a + b)⁻¹ := mul_pos (sq_pos_of_ne_zero ha.ne') (inv_pos.mpr hs)
    linarith [hrisk0, hr0_eval]
  · have hpos : 0 < b ^ 2 * (a + b)⁻¹ := mul_pos (sq_pos_of_ne_zero hb.ne') (inv_pos.mpr hs)
    linarith [hrisk1, hr1_eval]

/-- The tempered Cauchy–Schwarz precision denominator `(1 - α) + α * max (1 - a^2) τ₀` in
    `SRB-Spec` is strictly positive and uniformly bounded below by `min 1 τ₀`
    for all `α ∈ [0, 1]`. -/
theorem tempered_precision_denom_pos_and_bounded (α a τ₀ : ℝ)
    (h0 : 0 ≤ α) (h1 : α ≤ 1) (hτ : 0 < τ₀) :
    min 1 τ₀ ≤ (1 - α) + α * max (1 - a ^ 2) τ₀ ∧
      0 < (1 - α) + α * max (1 - a ^ 2) τ₀ := by
  have hm1 : min 1 τ₀ ≤ 1 := min_le_left 1 τ₀
  have hmt : min 1 τ₀ ≤ max (1 - a ^ 2) τ₀ := le_trans (min_le_right 1 τ₀) (le_max_right _ _)
  have hmin_pos : 0 < min 1 τ₀ := lt_min zero_lt_one hτ
  have hcomb : min 1 τ₀ ≤ (1 - α) + α * max (1 - a ^ 2) τ₀ := by
    have h1a : 0 ≤ 1 - α := sub_nonneg.mpr h1
    nlinarith
  exact ⟨hcomb, lt_of_lt_of_le hmin_pos hcomb⟩

end SignalDeconfoundingShrinkage

/-!
## Section B: Imperfect Proxy Covariate `(1 - r^2) * c^2 * v_W` Residual Floor Law
-/

section PartialProxyConfounder

/-- Residualizing a shared cross-family confounder `c * W` against an observable proxy covariate
    with squared correlation `r^2` reduces the confounder variance floor from `c^2 * v_W` to
    `(1 - r^2) * c^2 * v_W`, removing fraction `r^2 * c^2 * v_W`. -/
theorem partial_proxy_residual_confounder_variance (c r v_W : ℝ) :
    c ^ 2 * v_W - (1 - r ^ 2) * c ^ 2 * v_W = r ^ 2 * c ^ 2 * v_W := by
  ring

/-- When the observed covariate correlates imperfectly (`r^2 < 1`) with a nonzero shared
    confounder (`c ≠ 0`, `0 < v_W`), covariate residualization alone leaves a strict positive
    variance floor `(1 - r^2) * c^2 * v_W` plus the `B / F` family error budget. -/
theorem partial_proxy_strict_floor_when_imperfect (F : ℕ) (hF : 0 < F) (τ_sq σ_sq : Fin F → ℝ)
    (K_fam : Fin F → ℕ) (hK : ∀ f, 1 ≤ K_fam f) (hτ : ∀ f, 0 ≤ τ_sq f) (hσ : ∀ f, 0 ≤ σ_sq f)
    (B c r v_W : ℝ) (hB : ∀ f, τ_sq f + σ_sq f ≤ B) (hr : r ^ 2 < 1) (hc : c ≠ 0)
    (hvW : 0 < v_W) :
    let rem := (1 - r ^ 2) * c ^ 2 * v_W
    0 < rem ∧ rem ≤ rem + var_family_balanced F τ_sq σ_sq K_fam ∧
      rem + var_family_balanced F τ_sq σ_sq K_fam ≤ rem + B / (F : ℝ) := by
  dsimp only
  have hrpos : 0 < 1 - r ^ 2 := sub_pos.mpr hr
  have hrem_pos : 0 < (1 - r ^ 2) * c ^ 2 * v_W :=
    mul_pos (mul_pos hrpos (sq_pos_of_ne_zero hc)) hvW
  have hbal_nonneg : 0 ≤ var_family_balanced F τ_sq σ_sq K_fam := by
    unfold var_family_balanced
    refine mul_nonneg (by positivity) ?_
    refine Finset.sum_nonneg ?_
    intro f _
    have hKfpos : (0 : ℝ) ≤ (K_fam f : ℝ) := by exact_mod_cast (Nat.zero_le (K_fam f))
    exact add_nonneg (hτ f) (div_nonneg (hσ f) hKfpos)
  have hbal_le := var_family_balanced_le_inv_F F hF τ_sq σ_sq K_fam hK hσ B hB
  exact ⟨hrem_pos, by linarith, by linarith⟩

/-- The remaining confounder variance `(1 - r^2) * c^2 * v_W` decreases monotonically as
    proxy correlation strength `r^2` increases, vanishing exactly at `r^2 = 1`. -/
theorem partial_proxy_monotone_in_r_sq (c r₁ r₂ v_W : ℝ) (hv : 0 ≤ v_W) (hrr : r₁ ^ 2 ≤ r₂ ^ 2) :
    (1 - r₂ ^ 2) * c ^ 2 * v_W ≤ (1 - r₁ ^ 2) * c ^ 2 * v_W ∧
      (r₂ ^ 2 = 1 → (1 - r₂ ^ 2) * c ^ 2 * v_W = 0) := by
  constructor
  · have hcv : 0 ≤ c ^ 2 * v_W := mul_nonneg (sq_nonneg c) hv
    nlinarith
  · intro hr2; rw [hr2]; ring

end PartialProxyConfounder

/-!
## Section C: Multi-Factor `Rank-r` Spectral Subspace Perturbation & `O(n^{-1/2})` Rate
-/

section MultiFactorDavisKahan

/-- Multi-factor `rank-r` Rayleigh trace subspace perturbation bound: if top-`r` population
    eigenvalues satisfy `λ_r ≤ λ k` and `λ_next < λ_r`, empirical top-`r` directions capture
    eigenvalue sum `∑ k, λ k - ε * r` within `ε` trace perturbation (`Q_emp k ≤ Q_pop k + ε`)
    while escaping into orthogonal eigenspaces of eigenvalue at most `λ_next` with mass `w_sq k`,
    then total subspace leakage obeys `(λ_r - λ_next) * (∑ k, w_sq k) ≤ 2 * r * ε`. -/
theorem multifactor_rayleigh_trace_sin_sq_bound (r_dim : ℕ)
    (lam w_sq Q_pop Q_emp : Fin r_dim → ℝ) (lam_r lam_next ε : ℝ)
    (hgap : lam_next < lam_r)
    (hlam : ∀ k : Fin r_dim, lam_r ≤ lam k)
    (hw : ∀ k : Fin r_dim, 0 ≤ w_sq k)
    (hspec_pop :
      ∑ k : Fin r_dim, Q_pop k ≤ ∑ k : Fin r_dim, ((1 - w_sq k) * lam k + w_sq k * lam_next))
    (hopt_emp : ∑ k : Fin r_dim, (lam k - ε) ≤ ∑ k : Fin r_dim, Q_emp k)
    (hpert_emp : ∀ k : Fin r_dim, Q_emp k ≤ Q_pop k + ε) :
    let leak := ∑ k : Fin r_dim, w_sq k
    (lam_r - lam_next) * leak ≤ 2 * (r_dim : ℝ) * ε ∧
      leak ≤ (2 * (r_dim : ℝ) * ε) / (lam_r - lam_next) := by
  dsimp only
  set leak := ∑ k : Fin r_dim, w_sq k
  have hQ_emp_sum : ∑ k : Fin r_dim, Q_emp k ≤ (∑ k : Fin r_dim, Q_pop k) + (r_dim : ℝ) * ε := by
    calc ∑ k : Fin r_dim, Q_emp k
        ≤ ∑ k : Fin r_dim, (Q_pop k + ε) := Finset.sum_le_sum fun k _ => hpert_emp k
      _ = (∑ k : Fin r_dim, Q_pop k) + (r_dim : ℝ) * ε := by
            rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_fin, nsmul_eq_mul]
  have hopt_expand : (∑ k : Fin r_dim, lam k) - (r_dim : ℝ) * ε ≤ ∑ k : Fin r_dim, Q_emp k := by
    have heq : ∑ k : Fin r_dim, (lam k - ε) = (∑ k : Fin r_dim, lam k) - (r_dim : ℝ) * ε := by
      rw [Finset.sum_sub_distrib, Finset.sum_const, Finset.card_fin, nsmul_eq_mul]
    linarith [hopt_emp, heq]
  have hpop_expand :
      ∑ k : Fin r_dim, ((1 - w_sq k) * lam k + w_sq k * lam_next) =
        (∑ k : Fin r_dim, lam k) - ∑ k : Fin r_dim, (lam k - lam_next) * w_sq k := by
    have hpt : ∀ k : Fin r_dim,
        (1 - w_sq k) * lam k + w_sq k * lam_next = lam k - (lam k - lam_next) * w_sq k := by
      intro k; ring
    simp_rw [hpt, Finset.sum_sub_distrib]
  have hgap_sum : ∑ k : Fin r_dim, (lam k - lam_next) * w_sq k ≤ 2 * (r_dim : ℝ) * ε := by
    linarith
  have hlower : (lam_r - lam_next) * leak ≤ ∑ k : Fin r_dim, (lam k - lam_next) * w_sq k := by
    calc (lam_r - lam_next) * leak
        = ∑ k : Fin r_dim, (lam_r - lam_next) * w_sq k := by rw [Finset.mul_sum]
      _ ≤ ∑ k : Fin r_dim, (lam k - lam_next) * w_sq k :=
          Finset.sum_le_sum fun k _ => mul_le_mul_of_nonneg_right (by linarith [hlam k]) (hw k)
  have hprod : (lam_r - lam_next) * leak ≤ 2 * (r_dim : ℝ) * ε := le_trans hlower hgap_sum
  have hgappos : 0 < lam_r - lam_next := sub_pos.mpr hgap
  exact ⟨hprod, (le_div_iff₀ hgappos).mpr (by linarith)⟩

/-- For any fixed subspace rank `r_dim` and eigengap `γ_gap = λ_r - λ_{r+1} > 0`, the multi-factor
    Rayleigh trace bound decays to zero at rate `O(r_dim / sqrt(n))`. -/
theorem multifactor_subspace_rate_vanishes (r_dim : ℕ) (C γ_gap : ℝ) (hC : 0 ≤ C)
    (hgap : 0 < γ_gap) :
    Tendsto (fun n : ℕ => (2 * (r_dim : ℝ) * (C / Real.sqrt (n : ℝ))) / γ_gap) atTop (nhds 0) := by
  have hCr : 0 ≤ (r_dim : ℝ) * C := mul_nonneg (Nat.cast_nonneg r_dim) hC
  have hbase := davis_kahan_sample_rate_vanishes ((r_dim : ℝ) * C) γ_gap hCr hgap
  simpa [mul_assoc, mul_div_assoc] using hbase

/-- Specializing `multifactor_rayleigh_trace_sin_sq_bound` to rank `r_dim = 1` recovers
    the rank-1 Davis–Kahan bound `2 * ε / (λ₁ - λ₂)` identically. -/
theorem multifactor_reduces_to_rank_one (ε lam₁ lam₂ : ℝ) :
    (2 * ((1 : ℕ) : ℝ) * ε) / (lam₁ - lam₂) = 2 * ε / (lam₁ - lam₂) := by
  simp

end MultiFactorDavisKahan

/-!
## Section D: Continuous Measure-Theoretic TVD Event Witness & Partition Refinement
-/

section MeasureTheoreticTVD

/-- Continuous measure-theoretic variational TVD event witness attainment: on any general
    measurable space `(α, MeasurableSpace α)`, integrating the indicator of measurable event
    `S` under `P` and `Q` attains the exact set total-variation difference `|P.real S - Q.real S|`,
    and under probability measures obeys exact complement symmetry `P(S) - Q(S) = Q(Sᶜ) - P(Sᶜ)`. -/
theorem measure_tvd_event_witness_attainment {α : Type*} [MeasurableSpace α]
    (P Q : Measure α) [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    (S : Set α) (hS : MeasurableSet S) :
    |∫ _ in S, (1 : ℝ) ∂P - ∫ _ in S, (1 : ℝ) ∂Q| = |P.real S - Q.real S| ∧
      P.real S - Q.real S = Q.real Sᶜ - P.real Sᶜ := by
  constructor
  · simp [Measure.real]
  · have hP : P.real Sᶜ = 1 - P.real S := by
      simpa using (measureReal_compl (μ := P) hS)
    have hQ : Q.real Sᶜ = 1 - Q.real S := by
      simpa using (measureReal_compl (μ := Q) hS)
    linarith

/-- Discrete histogram TVD approximations of continuous distributions are monotonically
    non-decreasing under finite partition refinement: splitting each coarse cell `b` into
    sub-cells `j` can only increase or preserve total variation discrepancy. -/
theorem partition_tvd_refinement_monotone {B M : ℕ} (p_fine q_fine : Fin B → Fin M → ℝ) :
    (1 / 2 : ℝ) * ∑ b : Fin B, |((∑ j : Fin M, p_fine b j) - ∑ j : Fin M, q_fine b j)| ≤
      (1 / 2 : ℝ) * ∑ b : Fin B, ∑ j : Fin M, |p_fine b j - q_fine b j| := by
  refine mul_le_mul_of_nonneg_left ?_ (by positivity)
  refine Finset.sum_le_sum fun b _ => ?_
  have hsub : (∑ j : Fin M, p_fine b j) - ∑ j : Fin M, q_fine b j =
      ∑ j : Fin M, (p_fine b j - q_fine b j) := by
    rw [Finset.sum_sub_distrib]
  rw [hsub]
  exact Finset.abs_sum_le_sum_abs _ _

end MeasureTheoreticTVD
