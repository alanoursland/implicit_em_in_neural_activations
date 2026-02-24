# Where EM Lives in the Supervised Model

## The Precise Claim

Implicit EM arises if and only if:

1. Discrete alternatives exist.
2. Energies are assigned to each alternative.
3. Energies are exponentiated.
4. Likelihoods are normalized across alternatives.
5. The resulting objective is optimized via gradients.

These conditions are satisfied at specific locations in a network. Not everywhere. The question for our two-layer supervised model is: where exactly does EM live?

## The Standard Two-Layer MLP (Baseline)

```
x → Linear₁ → Softplus → Linear₂ → LayerNorm → CE(·, y)
```

### Output layer: EM is present.

Cross-entropy has LSE structure. The 10 class logits are alternatives. Softmax normalizes across them. The gradient is:

```
∂L/∂zⱼ = softmax(z)ⱼ − 𝟙[j = y]
```

Responsibilities exist. Competition exists. The label clamps the correct class to responsibility 1. Incorrect classes compete for the remaining mass. The output layer's weight matrix W₂ receives responsibility-weighted updates. This is implicit EM with supervision.

### Intermediate layer: EM is absent.

The 64 Softplus activations are not alternatives in competition. They are independent features. There is no normalization across them. The Jacobian of Softplus is diagonal:

```
∂aᵢ/∂zⱼ = σ(zⱼ) · δᵢⱼ     where σ is the sigmoid
```

No cross-unit terms. No competition. Unit 3's gradient does not depend on unit 7's activation. If both units help reduce the loss, both receive gradient. They may converge to identical features. Nothing prevents this.

The supervised gradient from CE does flow back to W₁. But by the time it arrives, it has been transformed through W₂ and the diagonal Softplus Jacobian. The intermediate layer sees "adjust these features to improve classification." It does not see "you have responsibility 0.3 for this input." The responsibility structure at the output does not induce responsibility structure at the intermediate layer.

### Summary: EM exists only at the output.

This is the standard situation described in where_em_lives.md. The output is a mixture model. The intermediate layer is a feature extractor trained by backprop through that mixture model. The feature extractor has no EM structure of its own.

## Our Two-Layer Model (Full ImplicitEM)

```
x → Linear₁ → Softplus → [aux loss] → NegLogSoftmin → Linear₂ → LayerNorm → CE(·, y)
```

### Output layer: EM is present (same as baseline).

Nothing changes at the output. CE + softmax over 10 classes. Label-clamped responsibilities. Responsibility-weighted updates to W₂.

### Intermediate layer: EM is now present.

The ImplicitEM layer introduces EM structure at the intermediate layer through two mechanisms.

**Mechanism 1: The auxiliary loss.**

The LSE auxiliary loss applied to distances d:

```
L_aux = -log Σ exp(-dⱼ)
```

This satisfies all five conditions. The 64 components are alternatives. Distances are energies. Exponentiation and normalization occur inside the LSE. The gradient of this loss with respect to each distance is exactly the responsibility:

```
∂L_aux/∂dⱼ = rⱼ = softmin(d)ⱼ
```

The auxiliary loss creates EM dynamics at the intermediate layer directly. Components compete for inputs. Responsibilities distribute gradient signal. W₁ receives responsibility-weighted updates from this loss. This is the same mechanism as the unsupervised model in Paper 2.

The volume control terms (variance + decorrelation) prevent the collapse that would otherwise occur, exactly as in the unsupervised case.

**Mechanism 2: The NegLogSoftmin Jacobian.**

Even without the auxiliary loss, NegLogSoftmin introduces competitive gradient structure into the backward pass. Its Jacobian is:

```
∂yᵢ/∂dⱼ = δᵢⱼ − rⱼ
```

This has off-diagonal terms: -rⱼ for i ≠ j. When the supervised gradient flows back from CE through W₂ and then through NegLogSoftmin, it encounters this Jacobian. The result: the gradient to dᵢ depends on all other components via the responsibility terms. Increasing dⱼ (pushing component j away) redistributes gradient to all other components.

This is competitive gradient structure. Not full EM — there is no explicit LSE objective at this layer — but the Jacobian introduces the coupling between units that ReLU and Softplus lack. The supervised signal, after passing through NegLogSoftmin, carries competition.

### Summary: EM exists at both layers.

The output layer has EM from cross-entropy (label-clamped). The intermediate layer has EM from the auxiliary loss (unsupervised) and competitive gradients from NegLogSoftmin. The two layers perform EM under different constraints — the output is supervised, the intermediate is unsupervised with volume control.

## The Six Ablation Configs, Mapped

| Config | Output EM | Intermediate EM | Intermediate Competition |
|--------|:---------:|:---------------:|:------------------------:|
| 1. Baseline MLP | ✓ (CE) | ✗ | ✗ (diagonal Jacobian) |
| 2. + NegLogSoftmin | ✓ (CE) | ✗ (no aux loss) | Partial (Jacobian only) |
| 3. + LSE only | ✓ (CE) | ✓ (but collapses) | ✓ |
| 4. + LSE + var | ✓ (CE) | ✓ (alive, redundant) | ✓ |
| 5. + LSE + var + tc | ✓ (CE) | ✓ (full) | ✓ |
| 6. + var + tc (no LSE) | ✓ (CE) | ✗ (no LSE) | Partial (Jacobian only) |

Config 2 is the interesting edge case. NegLogSoftmin provides competitive gradients via its Jacobian, but no explicit EM objective at the intermediate layer. The question: is the Jacobian alone sufficient to induce specialization? Or does it need the auxiliary LSE loss to create proper EM dynamics?

Config 6 is the other edge case. Volume control without LSE. Components stay alive and decorrelated, but is there mixture structure? Paper 2 found that without LSE, responsibility entropy drops — you get whitening, not clustering. The same should hold here.

## Two Sources of Gradient at W₁

The first layer's weights receive gradient from two paths:

**Path 1: Supervised (through the network)**
```
CE → ∂L/∂h → LayerNorm Jacobian → W₂ᵀ → NegLogSoftmin Jacobian → Softplus Jacobian → ∂L/∂W₁
```

This gradient wants W₁ to produce features useful for classification. It passes through the NegLogSoftmin Jacobian, which introduces competition. But the gradient's *purpose* is classification, not mixture modeling.

**Path 2: Auxiliary (direct)**
```
L_aux(d) → ∂L_aux/∂d → Softplus Jacobian → ∂L_aux/∂W₁
```

This gradient wants W₁ to produce well-formed mixture components. It is identical to the unsupervised gradient from Paper 2. Its purpose is mixture structure — explaining inputs, maintaining volume, preventing collapse.

Both paths update W₁ simultaneously. The total gradient is their sum (weighted by λ). The question is whether they cooperate or conflict.

**Expectation: Mostly cooperate.** A well-formed mixture over digit images naturally produces features useful for digit classification. Components that specialize to different digit variants are exactly what a classifier needs. The auxiliary loss produces structure; the supervised loss provides direction. They want compatible things.

**Possible tension:** The supervised loss might prefer a degenerate representation if it's easier to classify from. For example, mapping all inputs to a low-dimensional subspace that separates classes, ignoring within-class structure. The auxiliary loss would resist this because it demands all 64 components be alive and diverse. The balance depends on λ.

## What This Paper Tests

The central empirical question: does the intermediate layer need its own EM structure, or does the supervised gradient provide enough?

The theory says: the supervised gradient provides responsibility-weighted updates at the *output* layer. It does not provide them at the intermediate layer. ReLU and Softplus have diagonal Jacobians — no competition propagates backward. NegLogSoftmin's Jacobian introduces some competition, but without an explicit EM objective, there is no guarantee of proper mixture dynamics.

The ablation tests this directly. If the baseline MLP (no intermediate EM) produces degenerate intermediate representations while the full ImplicitEM layer produces structured ones — and if the ablation pattern matches Paper 2 — then intermediate EM matters independently of supervision.

## The Broader Point

In any deep supervised network:

- The output layer has EM structure from cross-entropy.
- Internal ReLU layers have no EM structure.
- Internal layers may be redundant, collapsed, or degenerate.
- Standard heuristics (BatchNorm, weight decay, dropout) provide implicit, unprincipled volume control.

The ImplicitEM layer provides explicit, principled EM structure at intermediate layers. The theory prescribes it. The experiment tests whether it matters.