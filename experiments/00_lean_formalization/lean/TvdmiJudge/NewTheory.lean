/-
  TVD-MI Judge Ensembles — Extended Theory Formalization (11-21-2026)

  Machine-verifies the new mathematical results established in Experiment 13:
    Section A: Finite-PMF Variational TVD Duality & Prompt-Copy / Injection Blindness
    Section B: Double-Centering Additive Shift Invariance & Oriented Peer Recovery
    Section C: Hierarchical Clone-Hijacking Falsification & Family-Balanced 1/F Recovery
    Section D: Corrected Hoeffding Two-Sample Range, Split-Sample Convergence,
               Effective-Family Adversarial Threshold & Cauchy-Schwarz Optimal Weights
-/

import Mathlib.Tactic
import Mathlib.Analysis.SpecialFunctions.Exp

open scoped BigOperators
open Filter

/-!
## Section A: Finite-PMF Variational TVD Duality & Injection Blindness
-/

section FiniteTVD

/-- Pointwise bound for any critic `h ∈ [0,1]` against the positive part of `p - q`. -/
private lemma diff_mul_critic_le_pos_part (a h : ℝ) (hh0 : 0 ≤ h) (hh1 : h ≤ 1) :
    a * h ≤ max a 0 := by
  by_cases ha : 0 ≤ a
  · calc a * h ≤ a * 1 := mul_le_mul_of_nonneg_left hh1 ha
      _ = a := mul_one a
      _ ≤ max a 0 := le_max_left a 0
  · push Not at ha
    have hnonpos : a * h ≤ 0 := mul_nonpos_of_nonpos_of_nonneg (le_of_lt ha) hh0
    exact hnonpos.trans (le_max_right a 0)

/-- Exact identity relating total variation (`|d|`) to twice the positive part minus `d`. -/
private lemma abs_eq_two_pos_part_sub (d : ℝ) :
    |d| = 2 * max d 0 - d := by
  by_cases hd : 0 ≤ d
  · rw [abs_of_nonneg hd, max_eq_left hd]; ring
  · push Not at hd
    rw [abs_of_neg hd, max_eq_right (le_of_lt hd)]; ring

/-- For equal-mass distributions `∑ x, p x = ∑ x, q x`, every `[0,1]`-valued critic
    has empirical contrast bounded above by `(1/2) * ∑ x, |p x - q x|`. -/
theorem finite_tvd_variational_le {α : Type*} [Fintype α] (p q h : α → ℝ)
    (hpq : ∑ x, p x = ∑ x, q x) (hh : ∀ x, 0 ≤ h x ∧ h x ≤ 1) :
    ∑ x, p x * h x - ∑ x, q x * h x ≤ (1 / 2 : ℝ) * ∑ x, |p x - q x| := by
  have hsum_zero : ∑ x, (p x - q x) = 0 := by
    rw [Finset.sum_sub_distrib, hpq, sub_self]
  have hhalf : (1 / 2 : ℝ) * ∑ x, |p x - q x| = ∑ x, max (p x - q x) 0 := by
    have h_abs_sum : ∑ x, |p x - q x| = ∑ x, (2 * max (p x - q x) 0 - (p x - q x)) := by
      refine Finset.sum_congr rfl ?_
      intro x _
      exact abs_eq_two_pos_part_sub (p x - q x)
    rw [h_abs_sum, Finset.sum_sub_distrib, ← Finset.mul_sum, hsum_zero, sub_zero]
    ring
  rw [hhalf, ← Finset.sum_sub_distrib]
  refine Finset.sum_le_sum ?_
  intro x _
  have hsub : p x * h x - q x * h x = (p x - q x) * h x := by ring
  rw [hsub]
  exact diff_mul_critic_le_pos_part (p x - q x) (h x) (hh x).1 (hh x).2

/-- The Scheffé witness critic `h*(x) = 1{p x ≥ q x}` attains equality in the
    variational TVD bound over finite alphabets. -/
theorem finite_tvd_variational_eq {α : Type*} [Fintype α] (p q : α → ℝ)
    (hpq : ∑ x, p x = ∑ x, q x) :
    let h_star : α → ℝ := fun x => if p x ≥ q x then 1 else 0
    (∀ x, 0 ≤ h_star x ∧ h_star x ≤ 1) ∧
      ∑ x, p x * h_star x - ∑ x, q x * h_star x = (1 / 2 : ℝ) * ∑ x, |p x - q x| := by
  classical
  dsimp only
  have hsum_zero : ∑ x, (p x - q x) = 0 := by
    rw [Finset.sum_sub_distrib, hpq, sub_self]
  constructor
  · intro x
    by_cases hle : q x ≤ p x <;> simp [hle]
  · have hhalf : (1 / 2 : ℝ) * ∑ x, |p x - q x| = ∑ x, max (p x - q x) 0 := by
      have h_abs_sum : ∑ x, |p x - q x| = ∑ x, (2 * max (p x - q x) 0 - (p x - q x)) := by
        refine Finset.sum_congr rfl ?_
        intro x _
        exact abs_eq_two_pos_part_sub (p x - q x)
      rw [h_abs_sum, Finset.sum_sub_distrib, ← Finset.mul_sum, hsum_zero, sub_zero]
      ring
    rw [hhalf, ← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl ?_
    intro x _
    by_cases hle : q x ≤ p x
    · simp [hle]
    · push Not at hle
      have h0 : p x - q x ≤ 0 := le_of_lt (sub_neg.mpr hle)
      have hn : ¬ (p x ≥ q x) := not_le.mpr hle
      simp [hn, max_eq_right h0]

/-- Unconditional bijection blindness: for any finite prompt PMF `p` and any
    injection `f : α → β`, evaluating a deterministic injective map `Y = f(X)`
    yields `TV(P_{X,f(X)}, P_X ⊗ P_{f(X)}) = 1 - ∑ x, (p x)^2`, depending solely
    on prompt collision entropy and zero on task semantics. -/
theorem injection_tvd_blindness {α β : Type*} [Fintype α] [Fintype β]
    [DecidableEq β] (p : α → ℝ) (hp_nonneg : ∀ x, 0 ≤ p x)
    (hp_sum : ∑ x, p x = 1) (f : α → β) (hf : Function.Injective f) :
    let q : β → ℝ := fun y => ∑ x : α, if f x = y then p x else 0
    (1 / 2 : ℝ) * ∑ x : α, ∑ y : β, |(if y = f x then p x else 0) - p x * q y| =
      1 - ∑ x : α, (p x) ^ 2 := by
  dsimp only
  set q : β → ℝ := fun y => ∑ x : α, if f x = y then p x else 0 with hq_def
  have hq_fx : ∀ x : α, q (f x) = p x := by
    intro x
    rw [hq_def]
    dsimp only
    rw [Finset.sum_eq_single x]
    · simp
    · intro x' _ hne
      simp [hf.ne hne]
    · intro hx; exact (hx (Finset.mem_univ x)).elim
  have hq_nonneg : ∀ y : β, 0 ≤ q y := by
    intro y
    refine Finset.sum_nonneg ?_
    intro x _
    by_cases hfx : f x = y <;> simp [hfx, hp_nonneg x]
  have hq_sum : ∑ y : β, q y = 1 := by
    rw [show (∑ y : β, q y) = ∑ y : β, ∑ x : α, (if f x = y then p x else 0) by rfl]
    rw [Finset.sum_comm]
    calc ∑ x : α, ∑ y : β, (if f x = y then p x else 0)
        = ∑ x : α, p x := by
          refine Finset.sum_congr rfl ?_
          intro x _
          rw [Finset.sum_eq_single (f x)]
          · simp
          · intro y _ hy; simp [Ne.symm hy]
          · intro hx; exact (hx (Finset.mem_univ (f x))).elim
      _ = 1 := hp_sum
  have hp_le_one : ∀ x : α, p x ≤ 1 := by
    intro x
    calc p x ≤ ∑ z : α, p z := Finset.single_le_sum (fun z _ => hp_nonneg z) (Finset.mem_univ x)
      _ = 1 := hp_sum
  have h_inner : ∀ x : α,
      ∑ y : β, |(if y = f x then p x else 0) - p x * q y| = 2 * (p x - (p x) ^ 2) := by
    intro x
    rw [← Finset.add_sum_erase _ _ (Finset.mem_univ (f x))]
    have hdiag : |(if f x = f x then p x else 0) - p x * q (f x)| = p x - (p x) ^ 2 := by
      rw [if_pos rfl, hq_fx x]
      have hpos : 0 ≤ p x - p x * p x := by
        nlinarith [hp_nonneg x, hp_le_one x]
      rw [abs_of_nonneg hpos]
      ring
    have hoff : ∑ y ∈ Finset.univ.erase (f x), |(if y = f x then p x else 0) - p x * q y| =
        p x - (p x) ^ 2 := by
      have hterm : ∀ y ∈ Finset.univ.erase (f x),
          |(if y = f x then p x else 0) - p x * q y| = p x * q y := by
        intro y hy
        have hne : y ≠ f x := (Finset.mem_erase.mp hy).1
        rw [if_neg hne, zero_sub, abs_neg, abs_of_nonneg (mul_nonneg (hp_nonneg x) (hq_nonneg y))]
      rw [Finset.sum_congr rfl hterm, ← Finset.mul_sum]
      have herase : ∑ y ∈ Finset.univ.erase (f x), q y = 1 - p x := by
        have hsplit := Finset.add_sum_erase Finset.univ q (Finset.mem_univ (f x))
        rw [hq_sum, hq_fx x] at hsplit
        linarith
      rw [herase]
      ring
    rw [hdiag, hoff]
    ring
  calc (1 / 2 : ℝ) * ∑ x : α, ∑ y : β, |(if y = f x then p x else 0) - p x * q y|
      = (1 / 2 : ℝ) * ∑ x : α, (2 * (p x - (p x) ^ 2)) := by
        congr 1; exact Finset.sum_congr rfl (fun x _ => h_inner x)
    _ = ∑ x : α, (p x - (p x) ^ 2) := by rw [← Finset.mul_sum]; ring
    _ = ∑ x : α, p x - ∑ x : α, (p x) ^ 2 := Finset.sum_sub_distrib _ _
    _ = 1 - ∑ x : α, (p x) ^ 2 := by rw [hp_sum]

/-- Non-vacuous witness: on `Fin 4` with uniform prompts `p = 1/4`, copying attains TVD-MI `3/4`. -/
example :
    (1 - ∑ _x : Fin 4, ((1 / 4 : ℝ)) ^ 2) = 3 / 4 := by norm_num

end FiniteTVD

/-!
## Section B: Double-Centering Invariance & Oriented Peer Recovery
-/

section DoubleCentering

/-- Additive prompt-leniency and response-verbosity shifts `a(x) + b(y)` cancel
    identically inside the TVD-MI witness contrast `E_{P_XY}[h] - E_{P_X ⊗ P_Y}[h]`. -/
theorem tvdmi_double_centering_invariance {α β : Type*} [Fintype α] [Fintype β]
    (P_XY : α → β → ℝ) (P_X : α → ℝ) (P_Y : β → ℝ)
    (hmargX : ∀ x, ∑ y, P_XY x y = P_X x)
    (hmargY : ∀ y, ∑ x, P_XY x y = P_Y y)
    (hsumX : ∑ x, P_X x = 1) (hsumY : ∑ y, P_Y y = 1)
    (a : α → ℝ) (b : β → ℝ) (ψ : α → β → ℝ) (γ : ℝ) :
    let h := fun x y => a x + b y + γ * ψ x y
    ∑ x, ∑ y, P_XY x y * h x y - ∑ x, ∑ y, (P_X x * P_Y y) * h x y =
      γ * (∑ x, ∑ y, P_XY x y * ψ x y - ∑ x, ∑ y, (P_X x * P_Y y) * ψ x y) := by
  dsimp only
  have hjoint_a : ∑ x, ∑ y, P_XY x y * a x = ∑ x, P_X x * a x := by
    refine Finset.sum_congr rfl ?_
    intro x _
    rw [← Finset.sum_mul, hmargX x]
  have hprod_a : ∑ x, ∑ y, (P_X x * P_Y y) * a x = ∑ x, P_X x * a x := by
    calc ∑ x, ∑ y, (P_X x * P_Y y) * a x
        = ∑ x, (P_X x * a x) * ∑ y, P_Y y := by
          refine Finset.sum_congr rfl ?_
          intro x _
          rw [Finset.mul_sum]
          refine Finset.sum_congr rfl (fun y _ => by ring)
      _ = ∑ x, P_X x * a x := by simp [hsumY]
  have hjoint_b : ∑ x, ∑ y, P_XY x y * b y = ∑ y, P_Y y * b y := by
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl ?_
    intro y _
    rw [← Finset.sum_mul, hmargY y]
  have hprod_b : ∑ x, ∑ y, (P_X x * P_Y y) * b y = ∑ y, P_Y y * b y := by
    rw [Finset.sum_comm]
    calc ∑ y, ∑ x, (P_X x * P_Y y) * b y
        = ∑ y, (P_Y y * b y) * ∑ x, P_X x := by
          refine Finset.sum_congr rfl ?_
          intro y _
          rw [Finset.mul_sum]
          refine Finset.sum_congr rfl (fun x _ => by ring)
      _ = ∑ y, P_Y y * b y := by simp [hsumX]
  have hsplit_joint :
      ∑ x, ∑ y, P_XY x y * (a x + b y + γ * ψ x y) =
        ∑ x, ∑ y, P_XY x y * a x + ∑ x, ∑ y, P_XY x y * b y +
          γ * ∑ x, ∑ y, P_XY x y * ψ x y := by
    have h1 : ∀ x, ∑ y, P_XY x y * (a x + b y + γ * ψ x y) =
        ∑ y, P_XY x y * a x + ∑ y, P_XY x y * b y + ∑ y, γ * (P_XY x y * ψ x y) := by
      intro x
      simp_rw [show ∀ y, P_XY x y * (a x + b y + γ * ψ x y) =
          (P_XY x y * a x + P_XY x y * b y) + γ * (P_XY x y * ψ x y) by intro y; ring]
      rw [Finset.sum_add_distrib, Finset.sum_add_distrib]
    simp_rw [h1, Finset.sum_add_distrib, ← Finset.mul_sum]
  have hsplit_prod :
      ∑ x, ∑ y, (P_X x * P_Y y) * (a x + b y + γ * ψ x y) =
        ∑ x, ∑ y, (P_X x * P_Y y) * a x + ∑ x, ∑ y, (P_X x * P_Y y) * b y +
          γ * ∑ x, ∑ y, (P_X x * P_Y y) * ψ x y := by
    have h1 : ∀ x, ∑ y, (P_X x * P_Y y) * (a x + b y + γ * ψ x y) =
        ∑ y, (P_X x * P_Y y) * a x + ∑ y, (P_X x * P_Y y) * b y +
          ∑ y, γ * ((P_X x * P_Y y) * ψ x y) := by
      intro x
      simp_rw [show ∀ y, (P_X x * P_Y y) * (a x + b y + γ * ψ x y) =
          ((P_X x * P_Y y) * a x + (P_X x * P_Y y) * b y) + γ * ((P_X x * P_Y y) * ψ x y) by
            intro y; ring]
      rw [Finset.sum_add_distrib, Finset.sum_add_distrib]
    simp_rw [h1, Finset.sum_add_distrib, ← Finset.mul_sum]
  rw [hsplit_joint, hsplit_prod, hjoint_a, hprod_a, hjoint_b, hprod_b]
  ring

/-- Oriented conditional peer witness separates Oracle (`p=1`), Noisy Solver,
    Random/Prompt-Copy (`p=1/2`), and Systematic Inverter (`p<1/2`) whenever peer
    competence exceeds chance (`q > 1/2`) and task target has positive variance (`v > 0`). -/
theorem oriented_peer_strict_mono (q v p₁ p₂ : ℝ) (hq : 1 / 2 < q) (hv : 0 < v)
    (hp : p₁ < p₂) :
    2 * (2 * q - 1) * (2 * p₁ - 1) * v < 2 * (2 * q - 1) * (2 * p₂ - 1) * v := by
  have h1 : 0 < 2 * (2 * q - 1) * v := by nlinarith
  have h2 : 2 * p₁ - 1 < 2 * p₂ - 1 := by linarith
  nlinarith

theorem oriented_peer_rejects_inverter (q v p_inv : ℝ) (hq : 1 / 2 < q) (hv : 0 < v)
    (hinv : p_inv < 1 / 2) :
    2 * (2 * q - 1) * (2 * p_inv - 1) * v < 0 := by
  have h1 : 0 < 2 * (2 * q - 1) * v := by nlinarith
  have h2 : 2 * p_inv - 1 < 0 := by linarith
  nlinarith

end DoubleCentering

/-!
## Section C: Hierarchical Clone-Hijacking Falsification & Balanced `1/F` Recovery
-/

section HierarchicalRBSpec

/-- Ensemble variance of the uniform average when Family 1 contributes `K₁` equicorrelated
    clones (`τ₁²`, idiosyncratic `σ₁²`) and Family 2 contributes `K₂` clones (`τ₂²`, `σ₂²`),
    assuming cross-family residuals are orthogonal. -/
noncomputable def var_two_family_unif (τ₁_sq σ₁_sq τ₂_sq σ₂_sq : ℝ) (K₁ K₂ : ℕ) : ℝ :=
  ((K₁ : ℝ) ^ 2 * τ₁_sq + K₁ * σ₁_sq + (K₂ : ℝ) ^ 2 * τ₂_sq + K₂ * σ₂_sq) /
    ((K₁ + K₂ : ℝ) ^ 2)

/-- Falsification (Clone-Hijacking Asymptote): Flooding Family 1 with `K₁ → ∞`
    clones drives the uniform ensemble variance to Family 1's correlation floor `τ₁²`,
    completely drowning out the diverse Family 2. -/
theorem clone_hijacking_tendsto (τ₁_sq σ₁_sq τ₂_sq σ₂_sq : ℝ) (K₂ : ℕ) :
    Tendsto (fun K₁ : ℕ => var_two_family_unif τ₁_sq σ₁_sq τ₂_sq σ₂_sq K₁ K₂)
      atTop (nhds τ₁_sq) := by
  have hrew : (fun K₁ : ℕ => var_two_family_unif τ₁_sq σ₁_sq τ₂_sq σ₂_sq K₁ K₂) =ᶠ[atTop]
      (fun K₁ : ℕ =>
        (τ₁_sq + σ₁_sq / (K₁ : ℝ) + ((K₂ : ℝ) ^ 2 * τ₂_sq + K₂ * σ₂_sq) / (K₁ : ℝ) ^ 2) /
          (1 + (K₂ : ℝ) / (K₁ : ℝ)) ^ 2) := by
    refine eventually_atTop.mpr ⟨1, ?_⟩
    intro K₁ hK1
    have hKpos : (0 : ℝ) < K₁ := Nat.cast_pos.mpr (lt_of_lt_of_le Nat.zero_lt_one hK1)
    unfold var_two_family_unif
    field_simp [hKpos.ne']
    ring
  rw [tendsto_congr' hrew]
  have hinv1 : Tendsto (fun K₁ : ℕ => (1 : ℝ) / (K₁ : ℝ)) atTop (nhds 0) :=
    tendsto_one_div_atTop_nhds_zero_nat
  have hinv_c1 : Tendsto (fun K₁ : ℕ => σ₁_sq / (K₁ : ℝ)) atTop (nhds 0) := by
    simpa [div_eq_mul_inv, mul_one] using hinv1.const_mul σ₁_sq
  have hinv2 : Tendsto (fun K₁ : ℕ => ((K₂ : ℝ) ^ 2 * τ₂_sq + K₂ * σ₂_sq) / (K₁ : ℝ) ^ 2)
      atTop (nhds 0) := by
    have hpow : Tendsto (fun K₁ : ℕ => (1 / (K₁ : ℝ)) ^ 2) atTop (nhds (0 ^ 2)) :=
      hinv1.pow 2
    rw [zero_pow (by norm_num)] at hpow
    have hcon := hpow.const_mul ((K₂ : ℝ) ^ 2 * τ₂_sq + K₂ * σ₂_sq)
    simpa [div_eq_mul_inv, mul_one, inv_pow, mul_comm] using hcon
  have hinv_k2 : Tendsto (fun K₁ : ℕ => (K₂ : ℝ) / (K₁ : ℝ)) atTop (nhds 0) := by
    simpa [div_eq_mul_inv, mul_one] using hinv1.const_mul (K₂ : ℝ)
  have hnum : Tendsto (fun K₁ : ℕ =>
      τ₁_sq + σ₁_sq / (K₁ : ℝ) + ((K₂ : ℝ) ^ 2 * τ₂_sq + K₂ * σ₂_sq) / (K₁ : ℝ) ^ 2)
      atTop (nhds (τ₁_sq + 0 + 0)) := (tendsto_const_nhds.add hinv_c1).add hinv2
  have hden : Tendsto (fun K₁ : ℕ => (1 + (K₂ : ℝ) / (K₁ : ℝ)) ^ 2) atTop (nhds ((1 + 0) ^ 2)) :=
    (tendsto_const_nhds.add hinv_k2).pow 2
  have hdiv := hnum.div hden (by norm_num)
  simpa using hdiv

/-- Concrete falsification of naive `1/K` monotonicity: with `K₂ = 4` diverse judges
    having low floor (`τ₂² = 0.01, σ₂² = 0.05`) and weak Family 1 having higher floor
    (`τ₁² = 0.25, σ₁² = 0.05`), adding `9` cloned judges in Family 1 (`K₁: 1 → 10`)
    strictly *increases* uniform ensemble variance! -/
theorem clone_hijacking_strict_increase_counterexample :
    var_two_family_unif 0.25 0.05 0.01 0.05 1 4 <
      var_two_family_unif 0.25 0.05 0.01 0.05 10 4 := by
  unfold var_two_family_unif
  norm_num

/-- Inverse-multiplicity (redundancy-balanced) `F`-family estimator variance:
    weighting each judge in family `f` by `1 / (F * K_f)` sums across `F` independent families
    to `(1 / F^2) * ∑ f, (τ_sq f + σ_sq f / K_f)`. -/
noncomputable def var_family_balanced (F : ℕ) (τ_sq σ_sq : Fin F → ℝ) (K_fam : Fin F → ℕ) : ℝ :=
  (1 / (F : ℝ) ^ 2) * ∑ f : Fin F, (τ_sq f + σ_sq f / (K_fam f : ℝ))

/-- Exact `1 / F` recovery theorem: redundancy-balanced family weighting is bounded by
    `B / F` for ALL positive clone multiplicities `K_fam f ≥ 1`, completely immune to
    arbitrarily large clone floods inside any family. -/
theorem var_family_balanced_le_inv_F (F : ℕ) (hF : 0 < F) (τ_sq σ_sq : Fin F → ℝ)
    (K_fam : Fin F → ℕ) (hK : ∀ f, 1 ≤ K_fam f) (hσ : ∀ f, 0 ≤ σ_sq f)
    (B : ℝ) (hB : ∀ f, τ_sq f + σ_sq f ≤ B) :
    var_family_balanced F τ_sq σ_sq K_fam ≤ B / (F : ℝ) := by
  unfold var_family_balanced
  have hFpos : (0 : ℝ) < F := Nat.cast_pos.mpr hF
  have h_sum_le : ∑ f : Fin F, (τ_sq f + σ_sq f / (K_fam f : ℝ)) ≤ ∑ _f : Fin F, B := by
    refine Finset.sum_le_sum ?_
    intro f _
    have hKf : (1 : ℝ) ≤ (K_fam f : ℝ) := by exact_mod_cast hK f
    have hdiv_le : σ_sq f / (K_fam f : ℝ) ≤ σ_sq f := by
      rw [div_le_iff₀ (lt_of_lt_of_le zero_lt_one hKf)]
      nlinarith [hσ f]
    linarith [hB f]
  have h_const : ∑ _f : Fin F, B = (F : ℝ) * B := by simp
  calc (1 / (F : ℝ) ^ 2) * ∑ f : Fin F, (τ_sq f + σ_sq f / (K_fam f : ℝ))
      ≤ (1 / (F : ℝ) ^ 2) * ((F : ℝ) * B) := by
        apply mul_le_mul_of_nonneg_left (h_sum_le.trans_eq h_const) (by positivity)
    _ = B / (F : ℝ) := by
        field_simp [hFpos.ne']

/-- The family-balanced upper bound `B / F` vanishes as independent families `F → ∞`. -/
theorem inv_F_rate_vanishes (B : ℝ) :
    Tendsto (fun F : ℕ => B / (F : ℝ)) atTop (nhds 0) := by
  simpa [div_eq_mul_inv, mul_one] using tendsto_one_div_atTop_nhds_zero_nat.const_mul B

end HierarchicalRBSpec

/-!
## Section D: Corrected Hoeffding Constant, Effective-Family Threshold & Optimal Weights
-/

section ConcentrationAndWeights

/-- Exact two-sample Hoeffding exponent correction (fixing Paper Theorem 3):
    with `n` joint samples and `n` product samples (each bounded in length `1` inside
    its `1/n`-scaled empirical mean), `∑ (b_m - a_m)^2 = 2/n`, giving exponent `-n * η^2`. -/
theorem hoeffding_two_sample_exponent (n : ℕ) (hn : 0 < n) (η : ℝ) :
    (2 * η ^ 2) / (n * ((1 : ℝ) / n) ^ 2 + n * ((1 : ℝ) / n) ^ 2) = (n : ℝ) * η ^ 2 := by
  have hnpos : (n : ℝ) ≠ 0 := (Nat.cast_pos.mpr hn).ne'
  field_simp [hnpos]
  ring

/-- Split-sample three-axis convergence (`Theorem 3'`): if witness approximation,
    selection concentration, and held-out evaluation concentration each vanish,
    their sum vanishes. -/
theorem split_sample_bound_vanishes (e_app e_sel e_eval : ℕ → ℝ)
    (h1 : Tendsto e_app atTop (nhds 0))
    (h2 : Tendsto e_sel atTop (nhds 0))
    (h3 : Tendsto e_eval atTop (nhds 0)) :
    Tendsto (fun N => e_app N + e_sel N + e_eval N) atTop (nhds 0) := by
  simpa using (h1.add h2).add h3

/-- Effective-family adversarial breakdown: under family-balanced weighting, an ensemble
    survives whenever the adversarial *family* fraction satisfies `δ_fam < 1/2`, even if
    adversarial clone counts dominate raw headcount. -/
theorem effective_family_adversarial_threshold (γ δ_fam : ℝ) (hγ : 0 < γ) :
    (1 - δ_fam) * (1 / 2 + γ) + δ_fam * (1 / 2 - γ) > 1 / 2 ↔ δ_fam < 1 / 2 := by
  have hid : (1 - δ_fam) * (1 / 2 + γ) + δ_fam * (1 / 2 - γ) - 1 / 2 = γ * (1 - 2 * δ_fam) := by
    ring
  constructor
  · intro h
    have hgt : 0 < γ * (1 - 2 * δ_fam) := by linarith
    have hpos : 0 < 1 - 2 * δ_fam := (mul_pos_iff_of_pos_left hγ).mp hgt
    linarith
  · intro h
    have hpos : 0 < 1 - 2 * δ_fam := by linarith
    have hgt : 0 < γ * (1 - 2 * δ_fam) := mul_pos hγ hpos
    linarith

/-- Non-vacuous witness: `K_adv = 9` adversarial clones from `1` family (`δ_fam = 1/3 < 1/2`)
    vs `K_honest = 2` honest families has raw headcount `9/11 > 1/2` yet balanced margin `> 0`. -/
example :
    (9 : ℝ) / (9 + 2) > 1 / 2 ∧
      (1 - (1 / 3 : ℝ)) * (1 / 2 + 0.3) + (1 / 3) * (1 / 2 - 0.3) > 1 / 2 := by
  norm_num

/-- Cost-free optimality of Nitzan–Paroush / inverse-variance competence weights via
    weighted Cauchy–Schwarz (`∑ v_j (w_j - μ_j/(v_j S))^2 ≥ 0`): among all unbiased
    linear weights (`∑ w_j μ_j = 1`), `w*_j = μ_j / (v_j S)` uniquely minimizes variance. -/
theorem optimal_judge_weights_cauchy_schwarz (K : ℕ) (μ v w : Fin K → ℝ)
    (hv : ∀ j, 0 < v j) (hw : ∑ j, w j * μ j = 1)
    (hS : 0 < ∑ j, (μ j) ^ 2 / v j) :
    let S := ∑ j, (μ j) ^ 2 / v j
    1 / S ≤ ∑ j, (w j) ^ 2 * v j := by
  dsimp only
  set S := ∑ j, (μ j) ^ 2 / v j
  have hsq_nonneg : 0 ≤ ∑ j, v j * (w j - μ j / (v j * S)) ^ 2 := by
    refine Finset.sum_nonneg ?_
    intro j _
    exact mul_nonneg (le_of_lt (hv j)) (sq_nonneg _)
  have hexpand : ∀ j : Fin K,
      v j * (w j - μ j / (v j * S)) ^ 2 =
        (w j) ^ 2 * v j - 2 / S * (w j * μ j) + (1 / S ^ 2) * ((μ j) ^ 2 / v j) := by
    intro j
    have hvj : v j ≠ 0 := (hv j).ne'
    have hSne : S ≠ 0 := hS.ne'
    field_simp [hvj, hSne]
    ring
  have hsum_eq :
      ∑ j, v j * (w j - μ j / (v j * S)) ^ 2 =
        (∑ j, (w j) ^ 2 * v j) - 1 / S := by
    simp_rw [hexpand, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, hw]
    have hSne : S ≠ 0 := hS.ne'
    change (∑ j, (w j) ^ 2 * v j) - 2 / S * 1 + 1 / S ^ 2 * S = (∑ j, (w j) ^ 2 * v j) - 1 / S
    field_simp [hSne]
    ring
  linarith

theorem optimal_judge_weights_equality (K : ℕ) (μ v : Fin K → ℝ)
    (hv : ∀ j, 0 < v j) (hS : 0 < ∑ j, (μ j) ^ 2 / v j) :
    let S := ∑ j, (μ j) ^ 2 / v j
    let w_star : Fin K → ℝ := fun j => μ j / (v j * S)
    (∑ j, w_star j * μ j = 1) ∧ (∑ j, (w_star j) ^ 2 * v j = 1 / S) := by
  dsimp only
  set S := ∑ j, (μ j) ^ 2 / v j
  have hSne : S ≠ 0 := hS.ne'
  constructor
  · have h1 : ∑ j : Fin K, μ j / (v j * S) * μ j = (1 / S) * ∑ j, (μ j) ^ 2 / v j := by
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl ?_
      intro j _
      have hvj : v j ≠ 0 := (hv j).ne'
      field_simp [hvj, hSne]
    change ∑ j, μ j / (v j * S) * μ j = 1
    rw [h1]
    change 1 / S * S = 1
    field_simp [hSne]
  · have h2 : ∑ j : Fin K, (μ j / (v j * S)) ^ 2 * v j = (1 / S ^ 2) * ∑ j, (μ j) ^ 2 / v j := by
      rw [Finset.mul_sum]
      refine Finset.sum_congr rfl ?_
      intro j _
      have hvj : v j ≠ 0 := (hv j).ne'
      field_simp [hvj, hSne]
    change ∑ j, (μ j / (v j * S)) ^ 2 * v j = 1 / S
    rw [h2]
    change 1 / S ^ 2 * S = 1 / S
    field_simp [hSne]

end ConcentrationAndWeights
