# The Conditioning Lemma: Uniform Curvature of LSE and Its Loss Under Composition

## Purpose

why_composition_breaks_em.md claims that implicit EM's optimization conditioning rests on the LSE objective having *parameter-independent bounded curvature*, and that composition through a learned linear map destroys this. This note states and proves the claim at the level of rigor needed for a paper. Empirical companions: supervised_study experiment 4 (curvature phase).

Everything here is elementary; the value is in assembling the pieces into the precise dichotomy the experiments test.

---

## Setup and Notation

Energies (distances) d ∈ ℝᴷ over K hypotheses. The LSE objective per sample:

```
L(d) = −log Σⱼ exp(−dⱼ)
```

Responsibilities r = softmax(−d), so rⱼ ≥ 0, Σⱼ rⱼ = 1.

For a vector v and distribution r on {1..K}, write Var_r(v) = Σⱼ rⱼvⱼ² − (Σⱼ rⱼvⱼ)² for the variance of the entries of v under r.

---

## Lemma 1 — Curvature of LSE in energy coordinates is uniformly bounded

**Claim.** ∇L(d) = r, ∇²L(d) = rrᵀ − diag(r), and

```
‖∇²L(d)‖₂ = λ_max(diag(r) − rrᵀ) ≤ ½    for every d ∈ ℝᴷ.
```

**Proof.** ∂L/∂dⱼ = exp(−dⱼ)/Σₖexp(−dₖ) = rⱼ, the gradient–responsibility identity. Differentiating again with ∂rⱼ/∂dₖ = −rⱼ(δⱼₖ − rₖ) gives ∇²L = rrᵀ − diag(r), which is negative semidefinite; its spectral norm is λ_max of M := diag(r) − rrᵀ ⪰ 0.

For any unit vector v: vᵀMv = Σⱼ rⱼvⱼ² − (Σⱼ rⱼvⱼ)² = Var_r(v). By Popoviciu's inequality, Var_r(v) ≤ ¼(maxⱼ vⱼ − minⱼ vⱼ)². For a unit vector, the max and min entries satisfy v_max² + v_min² ≤ 1, so (v_max − v_min)² ≤ 2(v_max² + v_min²) ≤ 2. Hence Var_r(v) ≤ ½. ∎

Two remarks. First, the bound is *uniform in d* — no matter where training goes, the curvature in energy coordinates never exceeds ½. Second, both the gradient (a point of the simplex, ‖r‖₂ ≤ 1) and the Hessian are bounded by absolute constants. Nothing about the parameters appears anywhere.

---

## Lemma 2 — Private linear energies inherit the bound, parameter-independently

**Setting.** dⱼ = wⱼᵀx + bⱼ: each energy has private parameters (row wⱼ), the load-bearing assumption. Let θ = (w₁, …, w_K) and consider L(θ) = L(d(θ)) for one sample x.

**Claim.**

```
‖∇_θ L‖₂ ≤ ‖x‖        and        ‖∇²_θ L‖₂ ≤ ½‖x‖²,
```

both uniform in θ. For the batch-mean loss, ‖∇²‖₂ ≤ ½·E‖x‖² — a constant of the data, invariant over training.

**Proof.** Since d is linear in θ there is no second-order term from the map θ ↦ d. The gradient w.r.t. wⱼ is rⱼx, so ‖∇_θL‖² = Σⱼ rⱼ²‖x‖² ≤ ‖x‖² (because Σⱼrⱼ² ≤ Σⱼrⱼ = 1). The Hessian has blocks (∇²_θL)ⱼₖ = Mⱼₖ · xxᵀ with M as in Lemma 1, i.e., ∇²_θL = M ⊗ xxᵀ up to sign; the spectral norm of a Kronecker product is the product of spectral norms, so ‖∇²_θL‖ ≤ ½‖x‖². Averaging over a batch preserves the bound with E‖x‖². ∎

**Corollary (training-invariant admissible step size).** The loss is β-smooth with β ≤ ½E‖x‖², so the descent lemma guarantees per-step decrease for any η < 2/β = 4/E‖x‖², *at every point of the trajectory*. The admissible learning-rate interval is fixed by the data before training starts and never shrinks. For MNIST, E‖x‖² ≈ 87.8, giving η* ≈ 0.046 under mean reduction.

Three honesty notes:
- **Sufficiency, not necessity.** β-smoothness gives a *sufficient* condition for stable descent; learning rates above the threshold are not forced to diverge. Paper 2 observed stability at lr = 0.1, above the worst-case bound. The lemma's explanatory content is the *invariance*: whatever learning rate empirically works at epoch 1 keeps working at epoch 100, because the bound never moves. That is what "learning-rate insensitivity" means operationally.
- **Reduction conventions matter for the constant.** Sum reduction over a batch of size B multiplies β by B. Only ratios and invariance survive convention changes; absolute thresholds must be quoted with the convention attached.
- **Kernels.** With d = φ(Wx + b) for a smooth monotone kernel φ (e.g. Softplus: φ′ ∈ (0,1), φ″ ≤ ¼), the chain rule adds a diagonal Jacobian and a second-order term Σⱼ rⱼφ″(zⱼ)xxᵀ, both bounded by absolute constants times ‖x‖²; the parameter-independence survives. ReLU satisfies the bounds a.e. (φ′ ∈ {0,1}, φ″ = 0) but introduces the absorbing-state pathology discussed in why_relu_breaks_em.md — a zeroth-order problem the curvature analysis does not see.

---

## Lemma 3 — Composition replaces the constant with σ_max(W₂)²

**Setting.** The supervised path of the Paper 3 model, LayerNorm omitted for clarity: logits h = W₂y with y = NLS(d), i.e., yⱼ = dⱼ + log Σₖ exp(−dₖ), and CE loss ℓ(h) = −log softmax(h)_c for label c. Consider the loss as a function of d.

**Claim.** With J := ∂y/∂d = I − 1rᵀ and p = softmax(h),

```
∇_d ℓ = Jᵀ W₂ᵀ (p − e_c)
∇²_d ℓ = Jᵀ W₂ᵀ (diag(p) − ppᵀ) W₂ J  +  (1ᵀW₂ᵀ(p − e_c)) · (diag(r) − rrᵀ)
```

and hence

```
‖∇_d ℓ‖ ≤ √2·(1 + √K)·σ_max(W₂)
‖∇²_d ℓ‖ ≤ ½(1 + √K)²·σ_max(W₂)²  +  (√(2K)/2)·σ_max(W₂).
```

**Proof sketch.** ∂logZ/∂d = −r gives J = I − 1rᵀ, with ‖J‖ ≤ 1 + ‖1‖‖r‖ ≤ 1 + √K. The CE Hessian w.r.t. logits is diag(p) − ppᵀ (Lemma 1 applied at the output), conjugated by the linear maps — the first term. The second term collects the curvature of the map d ↦ y: every coordinate of y shares the same second derivative ∇²logZ = diag(r) − rrᵀ, weighted by the sum of the incoming gradient 1ᵀW₂ᵀ(p − e_c), giving the stated bound via ‖p − e_c‖ ≤ √2 and Lemma 1. (LayerNorm inserts one more per-sample Jacobian, itself parameter- and input-dependent; it worsens the dependence and is omitted only to keep the display readable.) ∎

The constants are loose; the structure is the point. Every term now scales with σ_max(W₂) or its square, and W₂ is *learned*: it grows and rotates during training. The curvature at the intermediate layer is parameter-dependent, time-varying, and anisotropic with the eigenstructure of W₂ᵀ(·)W₂. The uniform ½ of Lemma 1 is gone.

---

## Proposition — The conditioning dichotomy

Combining the three lemmas:

1. **Private-parameter LSE** (Paper 2's setting): gradient uniformly bounded by ‖x‖, curvature uniformly bounded by ½E‖x‖². The admissible learning-rate interval is training-invariant, and there is no parameter-dependent anisotropy for an adaptive optimizer to exploit. Predictions: SGD learning-rate insensitivity within a data-determined range; no Adam advantage.

2. **The same objective behind a learned linear map** (Paper 3's setting): gradient and curvature at the intermediate layer scale with σ_max(W₂) and σ_max(W₂)² respectively. The admissible learning rate shrinks as W₂ grows, and curvature is anisotropic. Predictions: learning-rate sensitivity; Adam advantage; intermediate-layer curvature tracks σ_max(W₂)² over training while the LSE curvature stays below ½ throughout.

Prediction 2's curvature statement is directly measurable (power iteration on the exact Hessian w.r.t. d) and is the cleanest falsifiable form of "composition breaks EM conditioning."

**Status (experiment 4, supervised_study):** both predictions confirmed. The measured ‖∇²_d LSE‖ stays below ½ at every point of training in every arm and seed; under pure EM training it converges to 0.4993–0.4997 — the bound is *saturated exactly*, because responsibilities sharpen until two components tie at cluster boundaries (r = (½, ½, 0, …) achieves λ_max = ½). Distance from ½ is therefore a diagnostic of mixture sharpness, not just a worst case. The CE-path curvature at the same coordinates wanders over 6–17 (13–100× larger), non-monotonically, consistent with the σ_max(W₂)² scaling. See supervised_study/reports/4_composition_report.md.

---

## What the lemmas do not cover

- **The InfoMax terms.** Variance (−Σ log Var) and decorrelation penalties are batch-coupled and have unbounded curvature near degenerate configurations (Var → 0). The lemmas cover the LSE component only; empirically the InfoMax terms are small away from degeneracy, but the clean theory statement is about LSE.
- **ReLU's zeroth-order pathology.** Bounded curvature a.e. says nothing about the absorbing state at zero responsibility.
- **Stochastic gradients.** The descent lemma argument is stated for full-batch descent; the minibatch version needs the usual variance terms. The invariance argument is unaffected.
