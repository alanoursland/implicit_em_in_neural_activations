# Why Composition Breaks Implicit EM

## Context

Paper 2 found that the single-layer implicit EM model had an unusually well-conditioned optimization landscape: SGD was learning-rate insensitive across three orders of magnitude, Adam offered no advantage, and lower loss did not mean better features. Paper 3's optimizer experiment found that none of this survives when the EM layer is placed inside a supervised network. The interpretation there identified three mechanisms (W₂ᵀ scrambling, gradient magnitude interference, activation masking) but stopped at description.

This note gives the underlying theory: what invariant the EM conditioning rests on, exactly which operations preserve or destroy that invariant, and how each failure corresponds to a known regime where classical EM loses its guarantees. The conclusion is that composition is not a new failure mode — it is the neural incarnation of the oldest one.

---

## The Invariant: The Gradient Is a Conditional Expectation

For an LSE objective over energies,

```
L = −log Σⱼ exp(−dⱼ),     ∂L/∂dⱼ = rⱼ = softmin(d)ⱼ
```

the gradient with respect to the energy vector is not merely "a gradient." It is a posterior distribution — a point on the probability simplex. This gives two properties, and together they are the entire conditioning phenomenon.

### First order: the gradient is self-normalizing

Every entry satisfies 0 ≤ rⱼ ≤ 1, and Σⱼ rⱼ = 1 for every sample. The total gradient mass per sample is exactly one unit, divided among components by responsibility. Gradient magnitude cannot explode regardless of parameter scale, and components far from the data receive smoothly vanishing (never truncated) signal. This is why learning rate barely mattered in Paper 2.

### Second order: the curvature is universally bounded

The Hessian of LSE in energy coordinates is

```
∂²L/∂d² = diag(r) − r rᵀ
```

which satisfies 0 ⪯ H ⪯ ½·I at every point of the simplex (the Böhning bound from multinomial logistic regression). The bound is uniform in the parameters.

Now add the assumption that held in Paper 2: linear energies with **private parameters** — each component's energy dⱼ = wⱼᵀx depends only on its own row of W. Then the curvature with respect to the weights is bounded by

```
∇²_W L  ⪯  ½ · E[x xᵀ]  (blockwise)
```

a constant of the *data*, independent of the parameters, uniform over all of training. The landscape has one global smoothness constant. Any learning rate below a data-determined threshold converges, the threshold never moves during training, and there is no anisotropy for Adam to correct.

**One sentence: LSE with per-component private parameters has parameter-independent bounded curvature. That is what "EM conditioning" is.**

---

## Operations That Destroy the Invariant

Composition means the responsibility vector must travel through Jacobians before reaching parameters. Classify the operations by what they do to the simplex structure.

### 1. Dense, sign-indefinite, learned linear maps (the killer)

The supervised gradient reaching the intermediate layer is W₂ᵀ(p − y). Pulling a simplex vector back through an arbitrary matrix preserves nothing: non-negativity gone, normalization gone, per-component basis alignment gone.

The second-order damage is worse. The Hessian at the lower layer contains the conjugated term

```
W₂ᵀ (diag(p) − p pᵀ) W₂
```

so the curvature is now governed by the spectrum of W₂ — which is *learned*, changes throughout training, and is unbounded. The universal ½ has become ½·σ_max(W₂)², anisotropic across directions. This is precisely the ill-conditioning Adam exists to handle, which is why Adam became necessary in Paper 3.

Note what this exonerates: cross-entropy itself is fine. At the logit interface, (p − y) is a label-clamped responsibility — bounded, structured, EM-shaped. Both EM sites in the Paper 3 model were healthy. The structure died *in the corridor between them*.

### 2. Objective mixing

The total intermediate gradient is

```
∇_total = W₂ᵀ(p − y) + λ · r_aux
```

A convex-combination vector plus an arbitrary vector is an arbitrary vector. A sum of gradients is "a posterior of something" only if the total objective is itself the marginal likelihood of a single latent-variable model. CE + λ·LSE is not: it is two mixture models disagreeing about what the intermediate code means. Experiment 4 measured the disagreement at λ = 0.001: the CE path outweighs the EM path 30–70× (not the ~1000× report 3 inferred from λ alone), and the two gradients are near-orthogonal (cos ≈ 0). The EM signal is not opposed — it is drowned, a small orthogonal perturbation on a dominant signal. This is why NLS contributed nothing measurable in the ablation.

### 3. Hard gates (ReLU)

The pullback through ReLU is a {0,1} diagonal mask. In EM language this is hard assignment: responsibility exactly zero. Zero responsibility is an **absorbing state** — a component that stops receiving gradient can never argue its way back. This is not a new pathology; it is hard-EM's classical cluster-death problem (the same reason k-means empties clusters), surfacing as dead ReLU units. See why_relu_breaks_em.md; this note places it in the same taxonomy as the composition failures.

---

## Operations That Preserve the Invariant

The more useful list. Implicit EM structure survives pullback through:

**Per-coordinate monotone maps.** dⱼ → φ(dⱼ) with φ′ > 0 (Softplus, any kernel change). The pullback is a positive diagonal reweighting: sign structure, decoupling, and boundedness all survive. This is why kernel choice (Gaussian, Laplace, Student-t) is a free parameter of the framework — kernels are simplex-compatible; architectures generally are not.

**Parameter privacy.** Block-diagonal ∂d/∂θ — each hypothesis owns its parameters. This is the load-bearing assumption behind the bounded-curvature result, and the one composition silently violates.

**Stochastic (Markov) maps.** A matrix with non-negative entries and rows summing to 1 maps the simplex to itself; its transpose preserves non-negativity and total gradient mass under pullback (1ᵀAᵀg = (A1)ᵀg = 1ᵀg). Almost no standard layer is a Markov map — but softmax attention's value-mixing path *is* one. Responsibility structure survives pullback through attention weights in a way it cannot survive a dense FFN matrix. (Caveats: this is only the value path, only first order, and the attention matrix is input-dependent. But it suggests attention is the one standard architecture component that transports EM structure between layers rather than destroying it. Connects to relationship_to_attention.md.)

---

## The Same Failures Exist in Classical EM

Each broken invariant corresponds to a known regime where traditional EM loses its guarantees.

| Classical EM failure | Why EM fails there | Neural incarnation |
|---|---|---|
| **Shared parameters across components** | The Q-function no longer separates; the M-step is a coupled optimization; only "generalized EM" remains | Backprop composition: from the output mixture's view, the entire lower network is a parameter shared by all class-hypotheses |
| **Non-exponential-family complete-data model** | No sufficient statistics; the M-step has no closed form and is itself nonconvex | Component energies computed by a deep network; the "M-step" is SGD on a nonconvex inner problem |
| **Unbounded likelihood / singular components** | A component collapses onto a point, σ → 0, likelihood → ∞; needs the log-determinant to stay finite | LSE-only collapse (Paper 2); volume control is the neural log-det |
| **Hard assignment (hard-EM / k-means)** | Monotone convergence lost; zero-responsibility components can never recover | Dead ReLU units |
| **High missing information** | DLR rate result: convergence rate degrades as responsibilities approach uniform | Near-uniform responsibilities → vanishing competition; temperature controls this |

The central row is the first one. EM's efficiency comes from the E-step making the objective **separate over components** so the M-step decouples into independent per-component problems. That requires each component to own its parameters. The moment parameters are shared, the decomposition EM is made of no longer exists — and this is a structural fact about the algorithm, not something a regularizer can repair.

**Backprop composition is parameter sharing in disguise.** Paper 3 did not discover a new failure mode of implicit EM. It rediscovered, inside a neural network, the oldest structural limitation of the EM algorithm.

---

## At-Site Failures vs. Structural Failures

The table splits into two classes with different remedies:

**At-site failures** — collapse, hard assignment, uninformative responsibilities. These occur *at* the EM interface and are fixable there: volume control, soft kernels, temperature. This is why volume control transferred cleanly from Paper 2 to the supervised setting: it repairs a failure that lives at the site.

**Structural failures** — parameter sharing, non-exponential-family components. These break the decomposition EM consists of. No λ, kernel, or penalty can fix them, because the quantity being conditioned (a per-component posterior aligned with per-component parameters) no longer exists.

Composition is in the second class. This explains why λ-tuning in Paper 3 felt hopeless: it was an attempt to regularize a structural failure.

---

## Consequences

**The conditioning claim, stated precisely.** Implicit EM's optimization conditioning holds iff the responsibility gradient reaches parameters through simplex-compatible operations: per-component private parameters, per-coordinate monotone maps, and (partially) stochastic maps. It fails under sign-indefinite shared linear maps, objective mixing, and hard gates.

**A provable lemma.** The contrast between ∇²L ⪯ ½·E[xxᵀ] (private linear energies, parameter-independent) and the W₂ᵀ(diag(p) − ppᵀ)W₂ term (composition, spectrum-dependent) is short and self-contained. It converts Paper 3's empirical non-replication into a theorem about when replication is impossible.

**Each condition is separately testable.** (Status: experiment 4, supervised_study/reports/4_composition_report.md.)
- *Parameter privacy:* train W₁ with a stop-gradient so only the auxiliary EM loss reaches it (CE trains only the head). Prediction: Paper 2's anomalies reappear at W₁. **Confirmed** — probe-accuracy spread across a 1000× SGD lr range drops from 10.9 points (joint, λ=0.001) to 1.3 points (stop-gradient), and convergence-time invariance returns.
- *Objective mixing:* measure both gradient norms and vary λ. **Confirmed, with a refinement** — at λ=1 full connectivity behaves identically to stop-gradient: conditioning follows gradient *dominance*, not connectivity. The regime is graded by the CE/EM gradient ratio (measured 30–70× at λ=0.001, 0.02–0.05× at λ=1).
- *Hard gates:* ReLU vs. Softplus, already run (experiments 1–3).
- *Stochastic maps:* replace W₂ with a normalized non-negative map (or an attention read-out) and test whether partial conditioning survives. Speculative, high upside — still open.

**Implication for depth.** EM structure cannot be *inherited* through learned dense maps; it can only be *created locally* at each layer, with its own LSE objective and its own volume control, or transported through the narrow class of simplex-compatible maps. This is the theoretical justification for layer-wise implicit EM (layer_wise_implicit_em.md) — and a candidate explanation for why attention-based architectures behave differently from MLPs under the framework.
