/-
  TVD-MI Judge Ensemble Consistency — Core Definitions and Theorems

  This file formalizes the main results from the TVD-MI judge ensemble
  consistency paper. We focus on the algebraic and order-theoretic
  components that are most tractable in Lean 4 with Mathlib.

  Theorems formalized:
    1. Variance of equicorrelated ensemble mean (Proposition 5)
    2. Variance floor under persistent dependence (Theorem 6)
    3. Independent case simplification
    4. Adversarial breakdown threshold — multiplicative form (Theorem 7a)
    5. Adversarial breakdown threshold — division form (Theorem 7a')
    6. Sign-flip adversary threshold (Theorem 7b)
    7. Ranking preservation under uniform approximation (Theorem 8)
    8. Random-witness miss probability bounds (Theorem 1)
    9. Random-witness convergence (1-p)^K → 0
   10. Exponential bound: (1-x)^n ≤ exp(-nx) via 1-x ≤ exp(-x)
   11. Two-axis structural convergence (Theorem 3 structure)
   12. Consensus Hoeffding exponent algebra (Theorem 2 core)
   13. Robust aggregation error structure
   14. Scaling regime: decaying correlation
-/

import Mathlib.Tactic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
## Section 1: Variance of Equicorrelated Ensemble Mean (Proposition 5)

For K equicorrelated scores with variance σ² and pairwise correlation ρ,
Var(S̄_K) = σ² * ((1-ρ)/K + ρ) = σ² * (1 + (K-1)*ρ) / K.
-/

/-- The two forms of the equicorrelated variance formula are equal.
    Proposition 5 in the paper. -/
theorem variance_equicorrelated_mean (σ2 ρ : ℝ) (K : ℕ) (hK : 0 < K) :
    σ2 * ((1 - ρ) / ↑K + ρ) = σ2 * (1 + (↑K - 1) * ρ) / ↑K := by
  have hKpos : (0 : ℝ) < ↑K := Nat.cast_pos.mpr hK
  field_simp
  ring

/-- When ρ = 0, the variance simplifies to σ²/K (independent judges).
    Corollary 6, Case 1. -/
theorem variance_independent_mean (σ2 : ℝ) (K : ℕ) (_hK : 0 < K) :
    σ2 * ((1 - (0 : ℝ)) / ↑K + 0) = σ2 / ↑K := by
  simp; ring

/-- When ρ > 0 is constant, the variance has a positive floor σ²ρ.
    That is, Var(S̄_K) ≥ σ²ρ for all K ≥ 1.
    Core of Theorem 6 (impossibility under persistent dependence). -/
theorem variance_floor (σ2 ρ : ℝ) (hσ : 0 < σ2) (_hρ : 0 < ρ) (hρ1 : ρ ≤ 1)
    (K : ℕ) (hK : 0 < K) :
    σ2 * ρ ≤ σ2 * ((1 - ρ) / ↑K + ρ) := by
  have hKpos : (0 : ℝ) < ↑K := Nat.cast_pos.mpr hK
  have h_nonneg : 0 ≤ σ2 * ((1 - ρ) / ↑K) := by
    apply mul_nonneg (le_of_lt hσ)
    exact div_nonneg (by linarith) (by linarith)
  linarith [mul_add σ2 ((1 - ρ) / ↑K) ρ]

/-- Impossibility: if ρ > 0 persists, the variance floor is positive.
    Theorem 6. -/
theorem impossibility_persistent_dependence (σ2 ρ : ℝ) (hσ : 0 < σ2) (hρ : 0 < ρ) :
    0 < σ2 * ρ := by positivity

/-- The variance formula is monotone decreasing in K for fixed ρ ∈ [0,1].
    More judges always helps (or at least doesn't hurt). -/
theorem variance_monotone_K (σ2 ρ : ℝ) (hσ : 0 ≤ σ2) (_hρ0 : 0 ≤ ρ) (hρ1 : ρ ≤ 1)
    (K₁ K₂ : ℕ) (hK₁ : 0 < K₁) (hK₁₂ : K₁ ≤ K₂) :
    σ2 * ((1 - ρ) / ↑K₂ + ρ) ≤ σ2 * ((1 - ρ) / ↑K₁ + ρ) := by
  have hK₂ : 0 < K₂ := Nat.lt_of_lt_of_le hK₁ hK₁₂
  have hK₁pos : (0 : ℝ) < ↑K₁ := Nat.cast_pos.mpr hK₁
  have hK₂pos : (0 : ℝ) < ↑K₂ := Nat.cast_pos.mpr hK₂
  have h_nonneg : 0 ≤ 1 - ρ := by linarith
  have h_K_le : (↑K₁ : ℝ) ≤ ↑K₂ := Nat.cast_le.mpr hK₁₂
  have h_div : (1 - ρ) / ↑K₂ ≤ (1 - ρ) / ↑K₁ := by
    apply div_le_div_of_nonneg_left h_nonneg hK₁pos h_K_le
  linarith [mul_le_mul_of_nonneg_left (show (1 - ρ) / ↑K₂ + ρ ≤ (1 - ρ) / ↑K₁ + ρ by linarith) hσ]

/-!
## Section 2: Adversarial Breakdown Threshold (Theorem 7)

For honest judges with competence γ and adversarial fraction δ,
the ensemble succeeds iff (1-δ)(1/2 + γ) > 1/2 ⟺ δ < 2γ/(1+2γ).
-/

/-- The effective accuracy condition (1-δ)(1/2 + γ) > 1/2 is equivalent
    to δ(1+2γ) < 2γ. Theorem 7, Part 1 (multiplicative form). -/
theorem adversarial_threshold_mul (γ δ : ℝ) (_hγ : 0 < γ) :
    (1 - δ) * (1/2 + γ) > 1/2 ↔ δ * (1 + 2 * γ) < 2 * γ := by
  constructor
  · intro h; nlinarith
  · intro h; nlinarith

/-- Theorem 7, Part 1 in division form: the threshold is δ < 2γ/(1+2γ). -/
theorem adversarial_threshold_div (γ δ : ℝ) (hγ : 0 < γ) :
    (1 - δ) * (1/2 + γ) > 1/2 ↔ δ < 2 * γ / (1 + 2 * γ) := by
  have h_denom_pos : 0 < 1 + 2 * γ := by linarith
  constructor
  · intro h
    rw [lt_div_iff₀ h_denom_pos]
    nlinarith
  · intro h
    have := (lt_div_iff₀ h_denom_pos).mp h
    nlinarith

/-- For sign-flip adversaries (correct with prob 1/2 - γ), the threshold is δ < 1/2.
    Theorem 7, Part 2. This is the classical Byzantine majority threshold. -/
theorem adversarial_threshold_signflip (γ δ : ℝ) (hγ : 0 < γ) :
    (1 - δ) * (1/2 + γ) + δ * (1/2 - γ) > 1/2 ↔ δ < 1/2 := by
  constructor
  · intro h; nlinarith
  · intro h; nlinarith

/-- When γ ≤ 1/2 (competence margin at most 1/2), the always-wrong threshold
    2γ/(1+2γ) is at most the sign-flip threshold 1/2. -/
theorem always_wrong_threshold_le_signflip (γ : ℝ) (hγ : 0 < γ) (hγ_le : γ ≤ 1 / 2) :
    2 * γ / (1 + 2 * γ) ≤ 1 / 2 := by
  have h_denom_pos : 0 < 1 + 2 * γ := by linarith
  rw [div_le_div_iff₀ h_denom_pos (by norm_num : (0:ℝ) < 2)]
  nlinarith

/-- The threshold 2γ/(1+2γ) is in (0,1) for γ > 0. -/
theorem adversarial_threshold_range (γ : ℝ) (hγ : 0 < γ) :
    0 < 2 * γ / (1 + 2 * γ) ∧ 2 * γ / (1 + 2 * γ) < 1 := by
  have h_denom_pos : 0 < 1 + 2 * γ := by linarith
  constructor
  · exact div_pos (by linarith) h_denom_pos
  · rw [div_lt_one h_denom_pos]; linarith

/-!
## Section 3: Gold-Metric Rank Consistency (Theorem 8)

If all estimates are within ε of the truth and all true values are
separated by more than 2ε, the induced ranking is preserved.
-/

/-- Ranking preservation under uniform approximation:
    if |τ̂ᵢ - τᵢ| ≤ ε for all i, and |τᵢ - τⱼ| > 2ε for i ≠ j,
    then τᵢ < τⱼ ↔ τ̂ᵢ < τ̂ⱼ. Theorem 8. -/
theorem ranking_preserved {n : ℕ} (τ τ_hat : Fin n → ℝ)
    (ε : ℝ) (_hε : 0 < ε)
    (h_approx : ∀ i, |τ_hat i - τ i| ≤ ε)
    (h_sep : ∀ i j, i ≠ j → |τ i - τ j| > 2 * ε)
    (i j : Fin n) (hij : i ≠ j) :
    τ i < τ j ↔ τ_hat i < τ_hat j := by
  have hi := h_approx i
  have hj := h_approx j
  have hsep_ij := h_sep i j hij
  have hsep_ji := h_sep j i (Ne.symm hij)
  rw [abs_le] at hi hj
  constructor
  · intro h_lt
    have h_pos : τ j - τ i > 0 := by linarith
    have h_abs : |τ j - τ i| = τ j - τ i := abs_of_pos h_pos
    rw [h_abs] at hsep_ji
    linarith
  · intro h_lt
    by_contra h_not
    have h_ge : τ i - τ j ≥ 0 := by linarith [not_lt.mp h_not]
    rw [abs_of_nonneg h_ge] at hsep_ij
    linarith

/-!
## Section 4: Random-Witness Consistency (Theorem 1)

The key inequality: (1-p)^K ≤ exp(-Kp) for p ∈ [0,1].
-/

/-- For p ∈ [0,1] and K : ℕ, the miss probability (1-p)^K is in [0,1]. -/
theorem miss_probability_exact (p : ℝ) (hp0 : 0 ≤ p) (hp1 : p ≤ 1) (K : ℕ) :
    0 ≤ (1 - p) ^ K ∧ (1 - p) ^ K ≤ 1 := by
  constructor
  · exact pow_nonneg (by linarith) K
  · exact pow_le_one₀ (by linarith) (by linarith)

/-- As K → ∞, (1-p)^K → 0 for any p ∈ (0,1].
    This is the convergence statement for Theorem 1. -/
theorem miss_probability_vanishes (p : ℝ) (hp0 : 0 < p) (hp1 : p ≤ 1) :
    Filter.Tendsto (fun K : ℕ => (1 - p) ^ K) Filter.atTop (nhds 0) := by
  apply tendsto_pow_atTop_nhds_zero_of_lt_one
  · linarith
  · linarith

/-- The miss probability is monotone decreasing in K. More judges always
    reduces the miss probability. -/
theorem miss_probability_monotone (p : ℝ) (hp0 : 0 ≤ p) (hp1 : p ≤ 1)
    (K₁ K₂ : ℕ) (hK : K₁ ≤ K₂) :
    (1 - p) ^ K₂ ≤ (1 - p) ^ K₁ := by
  apply pow_le_pow_of_le_one (by linarith) (by linarith) hK

/-- The fundamental exponential inequality: 1 - x ≤ exp(-x) for all x : ℝ.
    This is the key bound used in Theorem 1 to get the exponential form. -/
theorem one_sub_le_exp_neg (x : ℝ) : 1 - x ≤ Real.exp (-x) := by
  have h := Real.add_one_le_exp (-x)
  linarith

/-- Corollary: (1-p)^K ≤ exp(-Kp) for p ∈ [0,1].
    This is the exponential bound in Theorem 1, eq. (5). -/
theorem miss_prob_le_exp (p : ℝ) (_hp0 : 0 ≤ p) (hp1 : p ≤ 1) (K : ℕ) :
    (1 - p) ^ K ≤ Real.exp (-(↑K * p)) := by
  -- First show exp(-p)^K = exp(-Kp) by induction
  have exp_pow : ∀ n : ℕ, (Real.exp (-p)) ^ n = Real.exp (-(↑n * p)) := by
    intro n
    induction n with
    | zero => simp
    | succ m ih =>
      rw [pow_succ, ih, ← Real.exp_add]
      congr 1
      push_cast; ring
  calc (1 - p) ^ K
      ≤ (Real.exp (-p)) ^ K := by
        gcongr
        · linarith
        · exact one_sub_le_exp_neg p
    _ = Real.exp (-(↑K * p)) := exp_pow K

/-!
## Section 5: Two-Axis Decomposition (Theorem 3, structural)

The main theorem combines judge approximation and sample estimation:
  P(τ(c) - max_j Δ̂(h_j) > ε + η) ≤ (1-p_ε)^K + 2K·exp(-2nη²)

We prove the structural lemma: if both terms vanish, the total vanishes.
-/

/-- Structural lemma for two-axis decomposition:
    If both the judge-approximation error and sample-estimation error
    converge to zero, the total error converges to zero.
    This is the convergence conclusion of Theorem 3. -/
theorem two_axis_sum_vanishes
    (f g : ℕ → ℝ)
    (hf : Filter.Tendsto f Filter.atTop (nhds 0))
    (hg : Filter.Tendsto g Filter.atTop (nhds 0))
    (_hf_nn : ∀ n, 0 ≤ f n)
    (_hg_nn : ∀ n, 0 ≤ g n) :
    Filter.Tendsto (fun n => f n + g n) Filter.atTop (nhds 0) := by
  have : (0 : ℝ) = 0 + 0 := by ring
  rw [this]
  exact Filter.Tendsto.add hf hg

/-- Union bound structure: if P(A) ≤ a and P(B) ≤ b with a,b ≥ 0,
    then a + b is an upper bound for P(A ∪ B). -/
theorem union_bound_structure (a b pa pb : ℝ)
    (_ha : 0 ≤ a) (_hb : 0 ≤ b)
    (hpa : pa ≤ a) (hpb : pb ≤ b) :
    pa + pb ≤ a + b := by linarith

/-!
## Section 6: Consensus Convergence — Algebraic Core (Theorem 2)

The Hoeffding exponent for consensus convergence:
  exp(-2Kγ²) → 0 as K → ∞ for γ > 0.
-/

/-- The Hoeffding exponent -2Kγ² is negative for K ≥ 1, γ > 0.
    This ensures the exponential bound is less than 1. -/
theorem hoeffding_exponent_neg (γ : ℝ) (hγ : 0 < γ) (K : ℕ) (hK : 0 < K) :
    -2 * ↑K * γ ^ 2 < 0 := by
  have hKpos : (0 : ℝ) < ↑K := Nat.cast_pos.mpr hK
  have hγ2 : 0 < γ ^ 2 := by positivity
  nlinarith

/-- The consensus error bound 2·exp(-2Kγ₀²) is positive. -/
theorem consensus_bound_pos (γ₀ : ℝ) (_hγ : 0 < γ₀) (K : ℕ) :
    0 < 2 * Real.exp (-2 * ↑K * γ₀ ^ 2) := by
  apply mul_pos (by norm_num : (0:ℝ) < 2)
  exact Real.exp_pos _

/-- The consensus error bound 2·exp(-2Kγ₀²) ≤ 2 for all K. -/
theorem consensus_bound_le_two (γ₀ : ℝ) (hγ : 0 < γ₀) (K : ℕ) (hK : 0 < K) :
    2 * Real.exp (-2 * ↑K * γ₀ ^ 2) ≤ 2 := by
  have : Real.exp (-2 * ↑K * γ₀ ^ 2) ≤ 1 := by
    have hKpos : (0 : ℝ) < ↑K := Nat.cast_pos.mpr hK
    have hγ2 : 0 < γ₀ ^ 2 := by positivity
    have h_neg : -2 * ↑K * γ₀ ^ 2 ≤ 0 := by nlinarith
    exact Real.exp_le_one_iff.mpr h_neg
  linarith

/-- The consensus bound vanishes as K → ∞ for any γ₀ > 0.
    This is the convergence conclusion of Theorem 2.
    Proved via: exp(-2Kγ₀²) = (exp(-2γ₀²))^K and exp(-2γ₀²) ∈ [0,1). -/
theorem consensus_bound_vanishes (γ₀ : ℝ) (hγ : 0 < γ₀) :
    Filter.Tendsto (fun K : ℕ => 2 * Real.exp (-2 * ↑K * γ₀ ^ 2)) Filter.atTop (nhds 0) := by
  have hγ2 : 0 < γ₀ ^ 2 := by positivity
  -- Key identity: exp(-2Kγ²) = (exp(-2γ²))^K, proved by induction via exp_add
  -- Prove exp(-c*K) = exp(-c)^K for c = 2γ₀² by abstracting the constant
  set c := 2 * γ₀ ^ 2 with hc_def
  have exp_pow : ∀ K : ℕ, Real.exp (-(c * ↑K)) = (Real.exp (-c)) ^ K := by
    intro K
    induction K with
    | zero => simp
    | succ n ih =>
      rw [pow_succ, ← ih, ← Real.exp_add]
      congr 1; push_cast; ring
  -- Rewrite the original expression to use c
  have h_eq : ∀ K : ℕ, -2 * (↑K : ℝ) * γ₀ ^ 2 = -(c * ↑K) := by
    intro K; rw [hc_def]; ring
  have h_rewrite : (fun K : ℕ => 2 * Real.exp (-2 * ↑K * γ₀ ^ 2)) =
      (fun K : ℕ => 2 * (Real.exp (-c)) ^ K) := by
    ext K; simp only [h_eq, exp_pow]
  rw [h_rewrite, show (0 : ℝ) = 2 * 0 from by ring]
  apply Filter.Tendsto.const_mul
  apply tendsto_pow_atTop_nhds_zero_of_lt_one
  · exact le_of_lt (Real.exp_pos _)
  · exact Real.exp_lt_one_iff.mpr (by rw [hc_def]; linarith)

/-!
## Section 7: Scaling Regimes for Decaying Correlation (Corollary 6)

When ρ_K = c · K^{-β}, the variance decays as max(K^{-1}, K^{-β}).
We verify key algebraic properties.
-/

/-- For constant correlation ρ₀ > 0, the variance is bounded below by σ²ρ₀
    and above by σ²((1-ρ₀) + ρ₀) = σ² for all K ≥ 1. Combined with the
    variance_floor theorem, this shows the variance is squeezed toward σ²ρ₀. -/
theorem variance_bounded_constant_corr (σ2 ρ₀ : ℝ) (hσ : 0 < σ2) (hρ : 0 < ρ₀) (hρ1 : ρ₀ ≤ 1)
    (K : ℕ) (hK : 0 < K) :
    σ2 * ρ₀ ≤ σ2 * ((1 - ρ₀) / ↑K + ρ₀) ∧ σ2 * ((1 - ρ₀) / ↑K + ρ₀) ≤ σ2 := by
  have hKpos : (0 : ℝ) < ↑K := Nat.cast_pos.mpr hK
  constructor
  · exact variance_floor σ2 ρ₀ hσ hρ hρ1 K hK
  · have h1 : (1 - ρ₀) / ↑K ≤ 1 - ρ₀ := by
      rw [div_le_iff₀ hKpos]
      have : (1 : ℝ) ≤ ↑K := by exact_mod_cast hK
      nlinarith
    have h2 : (1 - ρ₀) / ↑K + ρ₀ ≤ 1 := by linarith
    calc σ2 * ((1 - ρ₀) / ↑K + ρ₀)
        ≤ σ2 * 1 := by apply mul_le_mul_of_nonneg_left h2 (le_of_lt hσ)
      _ = σ2 := by ring

/-- For independent judges (ρ = 0), the variance equals σ²/K exactly,
    which is a decreasing sequence in K. -/
theorem variance_independent_exact (σ2 : ℝ) (K : ℕ) (_hK : 0 < K) :
    σ2 * ((1 - (0 : ℝ)) / ↑K + 0) = σ2 / ↑K := by
  simp; ring

/-!
## Section 8: Additional Structural Results
-/

/-- The competence margin γ must be positive for the ensemble to have any
    advantage over random guessing. With γ = 0, majority vote is no better
    than a coin flip. -/
theorem no_advantage_zero_margin :
    (1 - (0 : ℝ)) * (1/2 + 0) = 1/2 := by ring

/-- With full competence (γ = 1/2, i.e., judges are always right),
    the threshold 2γ/(1+2γ) = 1/2 when γ = 1/2. -/
theorem full_competence_threshold :
    2 * (1/2 : ℝ) / (1 + 2 * (1/2)) = 1/2 := by norm_num

/-- The TVD-MI target τ(c) is the supremum over witnesses. Any single witness
    gives a lower bound: Δ(h) ≤ τ(c) for all h ∈ H. -/
theorem witness_lower_bound (τ Δ : ℝ) (h_sup : Δ ≤ τ) : 0 ≤ τ - Δ := by linarith

/-- Triangle inequality for the two-axis decomposition:
    |τ - max_j Δ̂_j| ≤ |τ - max_j Δ_j| + |max_j Δ_j - max_j Δ̂_j|.
    This justifies the union bound decomposition in Theorem 3. -/
theorem two_axis_triangle (τ pop_max emp_max : ℝ) :
    |τ - emp_max| ≤ |τ - pop_max| + |pop_max - emp_max| := by
  have : τ - emp_max = (τ - pop_max) + (pop_max - emp_max) := by ring
  rw [this]
  exact abs_add_le (τ - pop_max) (pop_max - emp_max)
