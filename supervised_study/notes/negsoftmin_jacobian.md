# NegLogSoftmin Jacobian: Competitive Gradients in the Forward Pass

## The Jacobian

NegLogSoftmin maps distances d to calibrated distances y:

```
yⱼ = dⱼ + log Σₖ exp(-dₖ)
```

Differentiating yᵢ with respect to dⱼ:

```
∂yᵢ/∂dⱼ = δᵢⱼ + (1/Z) · ∂Z/∂dⱼ

         = δᵢⱼ + (1/Z) · (-exp(-dⱼ))

         = δᵢⱼ - rⱼ
```

where rⱼ = exp(-dⱼ) / Z is the responsibility of component j.

The full Jacobian is:

```
J = I - r1ᵀ
```

where r is the responsibility vector (column) and 1ᵀ is a row of ones. Every row of the Jacobian has the same off-diagonal terms: -rⱼ. The diagonal entries are 1 - rⱼ.

## What This Means

### Diagonal terms: 1 - rⱼ

The gradient of yⱼ with respect to its own input dⱼ is 1 - rⱼ. When component j has high responsibility (rⱼ ≈ 1), this term vanishes. The component is "saturated" — changes to its distance barely affect its calibrated output. When responsibility is low (rⱼ ≈ 0), the diagonal term is ≈ 1. The component passes gradient through almost unchanged.

This is self-regulating. Dominant components receive attenuated gradients. Weak components receive full gradients. The Jacobian automatically balances learning rates across components based on their current responsibility.

### Off-diagonal terms: -rⱼ

The gradient of yᵢ with respect to dⱼ (for i ≠ j) is -rⱼ. Changing component j's distance affects *every other component's* calibrated output. The effect is proportional to j's responsibility.

If component j has high responsibility and its distance increases (gets worse), all other components' calibrated distances decrease (get better). Responsibility redistributes. This is competition: one component's loss is every other component's gain.

### Contrast with Softplus (and ReLU)

Softplus Jacobian:

```
∂aᵢ/∂zⱼ = σ(zⱼ) · δᵢⱼ
```

Diagonal. No cross-unit terms. Unit i's gradient is independent of unit j. No competition.

ReLU Jacobian:

```
∂aᵢ/∂zⱼ = 𝟙[zⱼ > 0] · δᵢⱼ
```

Diagonal. Binary. No competition. Hard gating.

NegLogSoftmin is the first layer in our architecture that introduces off-diagonal Jacobian terms. It is where competition enters the backward pass.

## Competitive Gradient Flow

Consider the supervised gradient flowing back from CE through the network:

```
∂L/∂d = (∂L/∂y) · (∂y/∂d) = (∂L/∂y) · (I - r1ᵀ)
```

Let g = ∂L/∂y be the gradient arriving from the classification head. Then:

```
∂L/∂dᵢ = gᵢ - rᵢ · Σⱼ gⱼ    (wrong — let me redo)
```

Wait. The Jacobian acts on the left:

```
(∂L/∂d)ᵢ = Σⱼ (∂L/∂yⱼ) · (∂yⱼ/∂dᵢ) = Σⱼ gⱼ · (δⱼᵢ - rᵢ) = gᵢ - rᵢ · Σⱼ gⱼ
```

So:

```
∂L/∂dᵢ = gᵢ - rᵢ · ḡ
```

where ḡ = Σⱼ gⱼ is the sum of upstream gradients across all components.

This has a clean interpretation:

- **gᵢ**: the direct gradient to component i from the classification head. What the supervisor wants this component to do.
- **rᵢ · ḡ**: a correction term. Component i absorbs a share of the total gradient signal proportional to its responsibility.

If all upstream gradients are equal (ḡ/K for each), the correction term is rᵢ · ḡ and the net gradient is gᵢ - rᵢ · ḡ. Components with high responsibility get more correction. Components with low responsibility get less.

## The Centering Effect

The transformation ∂L/∂dᵢ = gᵢ - rᵢ · ḡ is a responsibility-weighted centering. It subtracts a baseline from each component's gradient. This is analogous to:

- **Advantage in RL:** A(s,a) = Q(s,a) - V(s). The value baseline centers the reward signal.
- **BatchNorm gradient:** Subtracts the mean gradient, centering updates.
- **Softmax gradient:** ∂L/∂zᵢ = (rᵢ - yᵢ) involves the same responsibility-weighted structure.

The centering ensures that gradients are *relative*, not absolute. A component's update depends not just on its own utility to the classifier, but on how that utility compares to the responsibility-weighted average. Components that are more useful than their responsibility warrants get positive net gradient. Components that are less useful get negative net gradient.

This is competition. Not imposed by an auxiliary loss. Emerging from the Jacobian of a calibration layer.

## Implications for Config 2

Config 2 in our ablation is NegLogSoftmin without any auxiliary loss:

```
d = Softplus(W₁x + b₁)
y = NegLogSoftmin(d)
h = LayerNorm(W₂y + b₂)
loss = CE(h, y_label)          # no auxiliary loss
```

The only gradient reaching W₁ comes from the supervised CE, flowing back through LayerNorm, W₂, and NegLogSoftmin. But that gradient passes through the Jacobian J = I - r1ᵀ. It acquires competitive structure.

**Prediction:** Config 2 should show *some* intermediate competition that the baseline MLP lacks. Not full EM dynamics (no explicit responsibility-weighted attraction toward data), but partial competition via the Jacobian. Specifically:

- Less redundancy than baseline (competition pushes components apart)
- Possibly some dead units (no variance penalty to prevent collapse)
- Some weight structure (competition encourages differentiation)
- But not the clean prototype structure of full ImplicitEM (no LSE attraction)

If this prediction holds, it means NegLogSoftmin is not a passive calibration layer. It actively shapes the optimization landscape by introducing competitive gradients. The auxiliary loss adds explicit EM dynamics on top of this inherent competition.

## Implications for Config 6

Config 6 has variance + decorrelation but no LSE:

```
loss = CE(h, y_label) + λ · (λ_var · L_var(d) + λ_tc · L_tc(d))
```

The NegLogSoftmin Jacobian still provides competitive gradients from the supervised path. The variance penalty keeps components alive. The decorrelation penalty keeps them diverse. But there is no explicit EM objective (no LSE loss).

**Prediction:** Config 6 should show alive, decorrelated components (from InfoMax) with some competitive structure (from Jacobian) but lower responsibility entropy than Config 5 (no LSE attraction). The representation is constrained to be healthy but not explicitly pushed toward mixture structure. Similar to Paper 2's "var + tc only" finding: whitening rather than clustering.

## The Jacobian Is Not Sufficient

The Jacobian provides competition but not attraction. It shapes how the supervised gradient distributes across components, but it does not pull components toward data they explain well. That attraction comes from the LSE auxiliary loss:

```
∂L_LSE/∂dⱼ = rⱼ
```

This gradient is always positive (pushing distances down for responsible components) and always proportional to responsibility. It is the M-step: move prototypes toward data in proportion to how much they explain.

The Jacobian redistributes whatever gradient arrives from downstream. The LSE loss generates gradient from the data directly. Both are needed for full EM dynamics:

- **Jacobian alone (Config 2):** Competition without attraction. Components differentiate but don't form prototypes.
- **LSE alone (Config 3):** Attraction without volume control. Components attract toward data but collapse.
- **Both (Config 5):** Full EM. Attraction + competition + volume control. Components form a proper mixture.

## Relationship to Softmax Jacobian

The NegLogSoftmin Jacobian J = I - r1ᵀ is closely related to the softmax Jacobian.

Softmax Jacobian: ∂rᵢ/∂zⱼ = rᵢ(δᵢⱼ - rⱼ)

NegLogSoftmin Jacobian: ∂yᵢ/∂dⱼ = δᵢⱼ - rⱼ

The NegLogSoftmin Jacobian is the softmax Jacobian divided by the responsibility rᵢ (row-wise). This means NegLogSoftmin passes *more* gradient to low-responsibility components than softmax would. Softmax attenuates both ways: low-responsibility components receive small gradients (rᵢ is small) and their effect on others is small (rⱼ is small). NegLogSoftmin attenuates only one way: effect on others is proportional to rⱼ, but the component's own gradient is not pre-multiplied by rᵢ.

This is a better gradient structure for learning. Low-responsibility components are not starved of gradient signal. They receive the full upstream gradient minus a baseline. They can recover. This is why NegLogSoftmin preserves EM-compatible dynamics while raw softmax can lead to dead units.

## Summary

NegLogSoftmin is not a passive transformation. Its Jacobian J = I - r1ᵀ introduces:

1. **Self-regulation:** Dominant components get attenuated gradients (diagonal: 1 - rⱼ).
2. **Competition:** Every component's gradient depends on all others (off-diagonal: -rⱼ).
3. **Centering:** Net gradient is gᵢ - rᵢ · ḡ. Updates are relative to a responsibility-weighted baseline.

These properties emerge from the calibration math. They are not designed or added. They are consequences of absorbing the partition function into the representation.

The Jacobian provides competitive gradient structure. The auxiliary loss provides EM dynamics. Together they make the ImplicitEM layer. Separately, each provides a partial picture — testable in Configs 2 and 6.