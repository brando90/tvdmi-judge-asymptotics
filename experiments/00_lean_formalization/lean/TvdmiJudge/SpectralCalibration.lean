/-
  TVD-MI Judge Ensembles — Spectral Perturbation, Affine Calibration, & Confounder Theory (11-21-2026)

  Machine-verifies Experiment 14 mathematical theorems (`11` theorems, zero `sorry`):
    Section A: Finite-Sample Bias-Variance / ICC(A,1) vs ICC(C,1) Gap & m-Anchor Calibration
    Section B: Cross-Family Shared Confounder Floor & Confounder Residualization (`CR-RB-Spec`)
    Section C: Finite-Dimensional Rank-1 Rayleigh / Davis-Kahan Spectral Perturbation & O(n^{-1/2}) Rate
-/

import Mathlib.Tactic
import Mathlib.Data.Real.Sqrt
import TvdmiJudge.NewTheory

open scoped BigOperators
open Filter

/-!
## Section A: Affine Calibration & Agreement vs Consistency ICC Gap
-/

section AffineCalibration

/-- Finite-sample MSE decomposes strictly into squared mean bias plus variance around the mean. -/
theorem mse_bias_variance_decomp {n : ℕ} (hn : 0 < n) (err : Fin n → ℝ) :
    let μ := (∑ i, err i) / (n : ℝ)
    (∑ i, (err i) ^ 2) / (n : ℝ) = μ ^ 2 + (∑ i, (err i - μ) ^ 2) / (n : ℝ) := by
  dsimp only
  set μ := (∑ i, err i) / (n : ℝ)
  have hnpos : (n : ℝ) ≠ 0 := (Nat.cast_pos.mpr hn).ne'
  have hsum_err : ∑ i, err i = (n : ℝ) * μ := by
    simp only [μ]
    field_simp [hnpos]
  have hexpand : ∑ i, (err i - μ) ^ 2 = (∑ i, (err i) ^ 2) - (n : ℝ) * μ ^ 2 := by
    have hpt : ∀ i : Fin n, (err i - μ) ^ 2 = (err i) ^ 2 - 2 * μ * err i + μ ^ 2 := by
      intro i; ring
    simp_rw [hpt, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum,
      Finset.sum_const, Finset.card_fin, nsmul_eq_mul, hsum_err]
    ring
  rw [hexpand]
  field_simp [hnpos]
  ring

/-- Debiasing a nonzero constant shift (`μ ≠ 0`) strictly reduces MSE from `(∑ err^2)/n`
    to variance `(∑ (err - μ)^2)/n`. -/
theorem strict_mse_reduction_from_debiasing {n : ℕ} (hn : 0 < n) (err : Fin n → ℝ)
    (hbias : (∑ i, err i) / (n : ℝ) ≠ 0) :
    (∑ i, (err i - (∑ j, err j) / (n : ℝ)) ^ 2) / (n : ℝ) < (∑ i, (err i) ^ 2) / (n : ℝ) := by
  have hdecomp := mse_bias_variance_decomp hn err
  dsimp only at hdecomp
  have hsq : 0 < ((∑ i, err i) / (n : ℝ)) ^ 2 := sq_pos_of_ne_zero hbias
  linarith

/-- Absolute agreement `ICC(A,1)` is always bounded above by consistency `ICC(C,1)` because
    squared mean bias `(μ_X - μ_Y)^2 ≥ 0` inflates the agreement denominator. -/
theorem icc_agreement_le_consistency (cov_XY v_X v_Y μ_X μ_Y : ℝ)
    (hcov : 0 ≤ cov_XY) (hv : 0 < v_X + v_Y) :
    (2 * cov_XY) / (v_X + v_Y + (μ_X - μ_Y) ^ 2) ≤ (2 * cov_XY) / (v_X + v_Y) := by
  have hsq : 0 ≤ (μ_X - μ_Y) ^ 2 := sq_nonneg _
  exact div_le_div_of_nonneg_left (by linarith) hv (by linarith)

/-- Strictly positive rank covariance combined with nonzero calibration bias (`μ_X ≠ μ_Y`)
    forces `ICC(A,1) < ICC(C,1)` strictly (explaining the uncalibrated `SummEval` gap). -/
theorem icc_agreement_lt_consistency_of_bias (cov_XY v_X v_Y μ_X μ_Y : ℝ)
    (hcov : 0 < cov_XY) (hv : 0 < v_X + v_Y) (hneq : μ_X ≠ μ_Y) :
    (2 * cov_XY) / (v_X + v_Y + (μ_X - μ_Y) ^ 2) < (2 * cov_XY) / (v_X + v_Y) := by
  have hsq : 0 < (μ_X - μ_Y) ^ 2 := sq_pos_of_ne_zero (sub_ne_zero.mpr hneq)
  exact div_lt_div_of_pos_left (by linarith) hv (by linarith)

/-- Once affine calibration matches means (`μ_X = μ_Y`), agreement `ICC(A,1)` equals
    consistency `ICC(C,1)` identically (and equals Pearson correlation when variances match). -/
theorem icc_agreement_eq_consistency_of_calibrated (cov_XY v_X v_Y μ : ℝ) :
    ((2 * cov_XY) / (v_X + v_Y + (μ - μ) ^ 2) = (2 * cov_XY) / (v_X + v_Y)) ∧
      (v_Y ≠ 0 → (2 * cov_XY) / (v_Y + v_Y + (μ - μ) ^ 2) = cov_XY / v_Y) := by
  constructor
  · simp
  · intro hvY
    simp only [sub_self, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, add_zero]
    field_simp [hvY]
    ring

/-- `m`-anchor mean-shift calibration excess MSE bound: `σ_cal^2 / m ≤ B / m` for `m ≥ 1`. -/
theorem anchor_calibration_excess_mse_le (B σ_cal_sq : ℝ) (hB : σ_cal_sq ≤ B)
    (m : ℕ) (hm : 0 < m) :
    σ_cal_sq / (m : ℝ) ≤ B / (m : ℝ) := by
  have hmpos : (0 : ℝ) < m := Nat.cast_pos.mpr hm
  exact div_le_div_of_nonneg_right hB hmpos.le

/-- `m`-anchor calibration excess error vanishes at rate `O(1/m)`. -/
theorem anchor_calibration_excess_tendsto_zero (B : ℝ) :
    Tendsto (fun m : ℕ => B / (m : ℝ)) atTop (nhds 0) := by
  simpa [div_eq_mul_inv, mul_one] using tendsto_one_div_atTop_nhds_zero_nat.const_mul B

end AffineCalibration

/-!
## Section B: Cross-Family Shared Confounder Variance Floor & Residualization
-/

section CrossFamilyConfounder

/-- Under a shared cross-family confounder `c * W` with `v_W = Var(W) > 0` and `c ≠ 0`,
    balancing clone multiplicities into `1/F` cannot remove the confounder floor `c^2 * v_W`. -/
theorem cross_family_confounder_variance_floor (F : ℕ) (hF : 0 < F) (τ_sq σ_sq : Fin F → ℝ)
    (K_fam : Fin F → ℕ) (hK : ∀ f, 1 ≤ K_fam f) (hτ : ∀ f, 0 ≤ τ_sq f) (hσ : ∀ f, 0 ≤ σ_sq f)
    (B c v_W : ℝ) (hB : ∀ f, τ_sq f + σ_sq f ≤ B) (hc : c ≠ 0) (hvW : 0 < v_W) :
    let tot_var := c ^ 2 * v_W + var_family_balanced F τ_sq σ_sq K_fam
    0 < c ^ 2 * v_W ∧ c ^ 2 * v_W ≤ tot_var ∧ tot_var ≤ c ^ 2 * v_W + B / (F : ℝ) := by
  dsimp only
  have hfloor : 0 < c ^ 2 * v_W := mul_pos (sq_pos_of_ne_zero hc) hvW
  have hbal_nonneg : 0 ≤ var_family_balanced F τ_sq σ_sq K_fam := by
    unfold var_family_balanced
    refine mul_nonneg (by positivity) ?_
    refine Finset.sum_nonneg ?_
    intro f _
    have hKfpos : (0 : ℝ) ≤ (K_fam f : ℝ) := by exact_mod_cast (Nat.zero_le (K_fam f))
    exact add_nonneg (hτ f) (div_nonneg (hσ f) hKfpos)
  have hbal_le := var_family_balanced_le_inv_F F hF τ_sq σ_sq K_fam hK hσ B hB
  exact ⟨hfloor, by linarith, by linarith⟩

/-- Confounder residualization (`CR-RB-Spec`): estimating `c_hat` shrinks the confounder
    floor to `(c - c_hat)^2 * v_W`, recovering the exact `B / F` rate when `c_hat = c`. -/
theorem confounder_residualization_recovery (F : ℕ) (hF : 0 < F) (τ_sq σ_sq : Fin F → ℝ)
    (K_fam : Fin F → ℕ) (hK : ∀ f, 1 ≤ K_fam f) (hσ : ∀ f, 0 ≤ σ_sq f)
    (B c c_hat v_W : ℝ) (hB : ∀ f, τ_sq f + σ_sq f ≤ B) :
    ((c - c_hat) ^ 2 * v_W + var_family_balanced F τ_sq σ_sq K_fam ≤
      (c - c_hat) ^ 2 * v_W + B / (F : ℝ)) ∧
      ((c - c) ^ 2 * v_W + var_family_balanced F τ_sq σ_sq K_fam ≤ B / (F : ℝ)) := by
  have hbal_le := var_family_balanced_le_inv_F F hF τ_sq σ_sq K_fam hK hσ B hB
  constructor
  · linarith
  · simp only [sub_self, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, zero_mul,
      zero_add]
    exact hbal_le

end CrossFamilyConfounder

/-!
## Section C: Finite-Dimensional Rank-1 Davis-Kahan Spectral Perturbation & O(n^{-1/2}) Rate
-/

section DavisKahan

/-- Finite-dimensional rank-1 Rayleigh / Davis–Kahan spectral perturbation bound:
    if unit vector `u₁` achieves population eigenvalue `λ₁`, any normalized candidate
    `u_hat = α • u₁ + w` (`α^2 + ∑ (w i)^2 = 1`) has population Rayleigh quotient bounded by
    `α^2 * λ₁ + (∑ (w i)^2) * λ₂` with eigengap `λ₁ - λ₂ > 0`, and `u_hat` beats `u₁`
    under an empirical form `Q_hat` within perturbation `ε` at both vectors, then
    `sin^2 θ = ∑ (w i)^2 ≤ 2 * ε / (λ₁ - λ₂)`. -/
theorem davis_kahan_rayleigh_sin_sq_bound (K : ℕ) (Q Q_hat : (Fin K → ℝ) → ℝ)
    (lam₁ lam₂ ε α : ℝ) (hgap : lam₂ < lam₁)
    (u₁ u_hat w : Fin K → ℝ)
    (hnorm : α ^ 2 + ∑ i : Fin K, (w i) ^ 2 = 1)
    (hspec_u1 : Q u₁ = lam₁)
    (hspec_uhat : Q u_hat ≤ α ^ 2 * lam₁ + (∑ i : Fin K, (w i) ^ 2) * lam₂)
    (hopt : Q_hat u₁ ≤ Q_hat u_hat)
    (hpert₁ : |Q_hat u₁ - Q u₁| ≤ ε)
    (hpert₂ : |Q_hat u_hat - Q u_hat| ≤ ε) :
    let sin_sq := ∑ i : Fin K, (w i) ^ 2
    (lam₁ - lam₂) * sin_sq ≤ 2 * ε ∧ sin_sq ≤ 2 * ε / (lam₁ - lam₂) := by
  dsimp only
  set sin_sq := ∑ i : Fin K, (w i) ^ 2
  have hlow : Q u₁ - ε ≤ Q_hat u₁ := by linarith [(abs_le.mp hpert₁).1]
  have hhigh : Q_hat u_hat ≤ Q u_hat + ε := by linarith [(abs_le.mp hpert₂).2]
  have hα : α ^ 2 = 1 - sin_sq := by linarith [hnorm]
  have hcomb : lam₁ - ε ≤ (1 - sin_sq) * lam₁ + sin_sq * lam₂ + ε := by
    calc lam₁ - ε = Q u₁ - ε := by rw [hspec_u1]
      _ ≤ Q_hat u₁ := hlow
      _ ≤ Q_hat u_hat := hopt
      _ ≤ Q u_hat + ε := hhigh
      _ ≤ α ^ 2 * lam₁ + sin_sq * lam₂ + ε := by linarith [hspec_uhat]
      _ = (1 - sin_sq) * lam₁ + sin_sq * lam₂ + ε := by rw [hα]
  have hprod : (lam₁ - lam₂) * sin_sq ≤ 2 * ε := by linarith
  have hgappos : 0 < lam₁ - lam₂ := sub_pos.mpr hgap
  exact ⟨hprod, (le_div_iff₀ hgappos).mpr (by linarith)⟩

/-- Combining Davis–Kahan `sin^2 θ ≤ 2 * ε / (λ₁ - λ₂)` with `O(n^{-1/2})` matrix estimation
    error `ε(n) = C / sqrt(n)` guarantees asymptotic eigenvector recovery `sin^2 θ → 0`. -/
theorem davis_kahan_sample_rate_vanishes (C γ_gap : ℝ) (hC : 0 ≤ C) (hgap : 0 < γ_gap) :
    Tendsto (fun n : ℕ => 2 * (C / Real.sqrt (n : ℝ)) / γ_gap) atTop (nhds 0) := by
  rw [Metric.tendsto_atTop]
  intro δ hδ
  by_cases hC0 : C = 0
  · refine ⟨1, fun n _ => ?_⟩
    simpa [hC0] using hδ
  · have hCpos : 0 < C := lt_of_le_of_ne hC (Ne.symm hC0)
    have hKpos : 0 < 2 * C / (γ_gap * δ) := by positivity
    obtain ⟨N, hN⟩ := exists_nat_gt ((2 * C / (γ_gap * δ)) ^ 2)
    refine ⟨max N 1, fun n hn => ?_⟩
    have hn1 : (1 : ℝ) ≤ (n : ℝ) := by
      exact_mod_cast le_trans (le_max_right N 1) hn
    have hnN : (N : ℝ) ≤ (n : ℝ) := by
      exact_mod_cast le_trans (le_max_left N 1) hn
    have hlt_n : (2 * C / (γ_gap * δ)) ^ 2 < (n : ℝ) := lt_of_lt_of_le hN hnN
    have h_sqrt_gt : 2 * C / (γ_gap * δ) < Real.sqrt (n : ℝ) :=
      (Real.lt_sqrt (by positivity)).2 hlt_n
    have h_sqrt_pos : 0 < Real.sqrt (n : ℝ) := Real.sqrt_pos.mpr (by linarith)
    have hval_nonneg : 0 ≤ 2 * (C / Real.sqrt (n : ℝ)) / γ_gap := by positivity
    have hbound : 2 * (C / Real.sqrt (n : ℝ)) / γ_gap < δ := by
      rw [div_lt_iff₀ hgap]
      have h1 : 2 * C < δ * γ_gap * Real.sqrt (n : ℝ) := by
        have := (div_lt_iff₀ (mul_pos hgap hδ)).mp h_sqrt_gt
        linarith
      calc 2 * (C / Real.sqrt (n : ℝ))
          = (2 * C) / Real.sqrt (n : ℝ) := by ring
        _ < δ * γ_gap := (div_lt_iff₀ h_sqrt_pos).mpr h1
    rw [Real.dist_eq, sub_zero, abs_of_nonneg hval_nonneg]
    exact hbound

end DavisKahan
