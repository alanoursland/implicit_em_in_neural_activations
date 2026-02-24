# Competition and Summary in the ImplicitEM Layer

## The Distinction

Competition and summary are different operations. Conflating them leads to incorrect architectures. The ImplicitEM layer implements both, in sequence, with clear boundaries.

**Competition:** Normalize across alternatives. Produce responsibilities. Create the zero-sum dynamic where one component's gain is another's loss. This is the E-step.

**Summary:** Collapse the result of competition into a representation that can be propagated forward. Preserve the information downstream layers need. This is what activations do.

These roles are not interchangeable. Softmin is a competition operator. An activation is a summary operator. Applying softmin where a summary is needed, or vice versa, produces pathological architectures.

## The Pipeline

The ImplicitEM layer implements the full pipeline described in competition_vs_summary.md:

```
1. Linear layer     →  z = Wx + b           (projection onto learned directions)
2. Kernel           →  d = Softplus(z)       (energy assignment)
3. Competition      →  r = softmin(d)        (responsibilities — implicit, via loss)
4. Summary          →  y = NegLogSoftmin(d)  (calibrated distances — forward pass)
```

Each step has a distinct role. None is redundant.

### Step 1: Projection

The linear layer projects the input onto K learned directions. Each row of W defines a component. The pre-activation zⱼ = wⱼᵀx + bⱼ is a signed scalar measuring where the input falls relative to component j's reference surface.

This step creates the hypothesis structure. K projections define K alternatives.

### Step 2: Kernel (Energy Assignment)

Softplus converts signed projections to non-negative distances:

```
dⱼ = log(1 + exp(zⱼ))
```

This is the logistic kernel. It assigns an energy to each hypothesis. Lower energy means better match. The choice of kernel encodes an inductive bias — logistic noise model, smooth distance, no flat regions.

This step creates the energy landscape. Each input has a distance to each component.

### Step 3: Competition

Competition is where responsibilities emerge. In the ImplicitEM layer, competition occurs through two mechanisms operating simultaneously.

**Mechanism A: The auxiliary LSE loss.**

```
L_LSE = -log Σ exp(-dⱼ)
```

The gradient of this loss with respect to each distance is exactly the responsibility:

```
∂L_LSE/∂dⱼ = rⱼ = exp(-dⱼ) / Σ exp(-dₖ)
```

This is explicit competition. The loss creates a softmin normalization over components. Responsibilities sum to one. Components compete for inputs. The gradient distributes learning signal in proportion to responsibility.

Responsibilities are never computed as a forward-pass quantity for propagation. They exist as gradients. The competition is implicit — embedded in the loss geometry.

**Mechanism B: The NegLogSoftmin Jacobian.**

The forward pass through NegLogSoftmin has Jacobian:

```
∂yᵢ/∂dⱼ = δᵢⱼ - rⱼ
```

When the supervised gradient flows backward through this Jacobian, it acquires competitive structure. Each component's gradient depends on all others via the -rⱼ terms. This is competition in the backward pass, independent of the auxiliary loss.

These two mechanisms are complementary. The auxiliary loss pulls components toward data they explain (attraction). The Jacobian redistributes the supervised gradient competitively (differentiation). Both involve responsibilities, but through different paths.

### Step 4: Summary

NegLogSoftmin produces the forward-pass output:

```
yⱼ = dⱼ + log Z
```

This is the summary step. It takes the raw distances — the energy landscape over which competition occurs — and produces a representation suitable for the next layer.

What does the summary preserve?

- **Relative distances.** yᵢ - yⱼ = dᵢ - dⱼ. The ranking of components is unchanged.
- **Probabilistic calibration.** exp(-yⱼ) = rⱼ. The absolute values encode responsibilities.
- **The distance convention.** Lower y means better match. The downstream layer receives distances, not probabilities or scores.

What does the summary add?

- **The partition function.** log Z is absorbed into each output. The downstream layer doesn't need to renormalize.
- **Competitive gradient structure.** The Jacobian couples all components, as described above.

What does the summary discard?

- **Nothing.** NegLogSoftmin is invertible (given y, you can recover d by subtracting the mean shift). No information is lost. This is crucial. Unlike softmax (which maps to the simplex and loses magnitude) or ReLU (which zeros out half the signal), NegLogSoftmin preserves all information in the distances while adding calibration.

## Why Not Softmin as the Forward Pass?

A natural but incorrect idea: use softmin (the responsibilities themselves) as the representation passed to the next layer.

```
rⱼ = exp(-dⱼ) / Σ exp(-dₖ)      # softmin output
```

Problems:

**Magnitude loss.** Responsibilities lie on the simplex. They sum to one. If d = [0.1, 5.0, 5.0], the responsibilities are approximately [0.99, 0.005, 0.005]. If d = [0.1, 50.0, 50.0], the responsibilities are approximately [1.0, 0.0, 0.0]. Both say "component 1 wins" but the second case is far more certain. The distinction — which the raw distances carry — is lost.

**Gradient starvation.** The softmax Jacobian is ∂rᵢ/∂dⱼ = rᵢ(rⱼ - δᵢⱼ). For a component with rᵢ ≈ 0, the gradient is pre-multiplied by rᵢ ≈ 0. Dead components receive no gradient and cannot recover. This is the same problem as ReLU dead units, in a different form.

**Simplex constraint.** The output is constrained to a (K-1)-dimensional simplex. The downstream linear layer operates on this simplex. Representational capacity is reduced from K dimensions to K-1.

NegLogSoftmin avoids all three problems. Magnitude is preserved (the shift is uniform). Low-responsibility components receive full gradient (the Jacobian diagonal is 1 - rⱼ ≈ 1). The output lives in full K-dimensional space.

## Why Not Raw Distances as the Forward Pass?

Another natural idea: skip NegLogSoftmin entirely. Pass the raw Softplus distances d to the next layer.

```
d = Softplus(Wx + b)             # just pass these forward
```

This is Config 1 (baseline MLP). Problems:

**No calibration.** Distance magnitudes are arbitrary. A network with large weights produces large distances. A network with small weights produces small distances. The downstream layer must learn to compensate for whatever scale the upstream layer produces. Scale coupling between layers.

**No competitive gradients.** The Softplus Jacobian is diagonal. No cross-unit terms. The supervised gradient passes through without acquiring competitive structure. Components are independent. Redundancy is unconstrained.

**No probabilistic semantics.** The number 3.7 means nothing without the partition function. Is it large or small relative to other components? For this input or in general? The downstream layer receives numbers without context.

NegLogSoftmin provides calibration, competitive gradients, and probabilistic semantics. It is the minimal operation that converts raw distances into a proper posterior summary.

## The Complete Picture

```
         Competition                          Summary
         (implicit, via loss + Jacobian)      (explicit, via forward pass)
              │                                    │
              ▼                                    ▼
x → W₁x+b₁ → Softplus → ─── d ─── → NegLogSoftmin → y → next layer
                               │
                               ▼
                          aux loss (LSE + var + decorr)
                          creates responsibilities as gradients
                          provides EM dynamics + volume control
```

The raw distances d are the central object. They sit at the junction of competition and summary.

Looking left: the auxiliary loss operates on d, creating EM dynamics. Responsibilities emerge as gradients of this loss. Components compete. Volume control prevents collapse.

Looking right: NegLogSoftmin transforms d into y, the calibrated representation. The Jacobian introduces competitive gradient structure. The output carries probabilistic semantics.

Looking down: the linear layer and Softplus produce d from inputs. Their parameters W₁, b₁ receive gradients from both paths — the auxiliary loss and the supervised loss flowing back through the summary.

The distances d are where competition happens. The calibrated distances y are the summary of that competition. The distinction is maintained throughout.

## Relationship to Classical EM

In classical EM on a Gaussian mixture model:

```
E-step:  Compute responsibilities rⱼ = P(j|x) for each component     → Competition
M-step:  Update μⱼ, Σⱼ using responsibility-weighted statistics       → Parameter update
Output:  Cluster assignments, mixture parameters                       → Summary
```

In the ImplicitEM layer:

```
Forward:   Compute distances dⱼ = Softplus(wⱼᵀx + bⱼ)                → Energy assignment
Implicit E: Responsibilities rⱼ appear as gradients of LSE loss        → Competition
Implicit M: W₁ updated by responsibility-weighted gradients             → Parameter update
Summary:   Calibrated distances yⱼ = dⱼ + log Z passed forward         → Summary
```

The correspondence is direct. The E-step is the auxiliary loss gradient. The M-step is the parameter update. The summary is NegLogSoftmin. Competition and summary are distinct in both classical EM and in the ImplicitEM layer.

The difference: in classical EM, the E-step is an explicit computation. In implicit EM, it is embedded in the loss geometry. The responsibilities are never stored — they exist as gradients during backpropagation and are consumed by the parameter update. The summary (NegLogSoftmin) does not need access to the responsibilities because it operates on the distances directly.

## Why This Matters

Getting the distinction right prevents two errors:

**Error 1: Using softmin as an activation.** This conflates competition with summary. Softmin is for deciding responsibilities, not for producing representations. The result is magnitude loss, gradient starvation, and simplex constraints.

**Error 2: Omitting competition entirely.** This is the baseline MLP. Raw distances pass through without normalization. No competition, no EM, no specialization pressure. Components are free to be redundant.

The ImplicitEM layer avoids both errors by assigning each operation its correct role. Competition lives in the loss and the Jacobian. Summary lives in NegLogSoftmin. Distances mediate between them.