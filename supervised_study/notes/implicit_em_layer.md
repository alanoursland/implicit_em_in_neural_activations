# ImplicitEM Layer: Supervised Regime of Implicit EM Theory

## Background

### The Theory (Paper 1: Gradient Descent as Implicit EM)

For any objective with log-sum-exp structure over distances, the gradient with respect to each distance is exactly the negative posterior responsibility:

$$\partial L / \partial d_j = -r_j$$

This identity holds across three regimes:
- **Unsupervised:** Responsibilities fully latent. Components compete freely.
- **Conditional (Attention):** Responsibilities conditioned on queries.
- **Constrained (Supervised):** Labels clamp one responsibility to 1.

The theory also identifies a failure mode: neural LSE objectives lack the log-determinant term that prevents collapse in Gaussian mixture models. Without volume control, representations degenerate.

### The Unsupervised Validation (Paper 2: Decoder-Free SAEs)

Paper 2 tested the unsupervised regime. Built exactly what the theory prescribed:
- Single linear layer + ReLU (distances)
- LSE loss (EM dynamics)
- InfoMax regularization: variance + decorrelation (volume control)

All predictions confirmed. LSE alone collapses. Variance prevents dead units. Decorrelation prevents redundancy. Features are interpretable mixture components.

### This Paper: The Supervised Regime

Paper 2 validated the unsupervised case. This paper validates the supervised case.

The question: Can implicit EM theory prescribe a supervised architecture? What role does volume control play when labels are present?

---

## Key Theoretical Points

### Single-Layer CE Already Has Volume Control

In standard cross-entropy classification with K classes:
- Labels clamp each class to responsibility 1 for its examples
- Every class receives gradient signal → no dead classes (anti-collapse)
- Classes must separate to minimize loss → no redundancy (anti-redundancy)

Supervision provides implicit volume control at the output layer. This is why single-layer CE works without any regularization beyond the labels themselves. The labels do the work that variance + decorrelation penalties do in the unsupervised case.

This should be demonstrated briefly, not belabored.

### Intermediate Layers Lack Volume Control

In a multi-layer supervised network, labels only clamp the output. The supervised gradient flows back to intermediate layers, but it only demands "produce something the output layer can classify." It is indifferent to the geometric health of intermediate representations.

Intermediate layers can collapse, become redundant, or degenerate — as long as the output layer can still read the result. This is the unsupervised volume control problem, occurring inside a supervised network.

This explains why deep networks need BatchNorm, LayerNorm, residual connections, weight decay — they all provide implicit volume control for intermediate representations. The theory reframes these as heuristic solutions to a principled problem.

### The Prescription

The theory prescribes: every layer that computes distances and participates in implicit EM needs volume control. For intermediate layers in supervised networks, this must be provided explicitly, because labels only control the output.

---

## The ImplicitEM Layer

### Definition

A single computational unit that implements all three requirements of implicit EM theory:

1. **Distance computation:** Linear + Softplus → non-negative distances
2. **Volume control:** LSE + variance + decorrelation (auxiliary loss on distances)
3. **Calibration:** NegLogSoftmin → calibrated distances (proper probabilistic representation)

### Distance Computation

```
d = Softplus(Wx + b)
```

Softplus (not ReLU) because:
- Always positive (proper distances)
- Smooth (no flat region at zero)
- No half-space degeneracy where entire regions map to distance zero

Each component measures how far the input is from a learned prototype surface. Lower = better match.

### Volume Control (Auxiliary Loss)

Applied to raw distances d. Same combined_loss as unsupervised paper:

```
L_aux = L_LSE + λ_var · L_var + λ_tc · L_tc
```

Where:
- L_LSE = -log Σ exp(-d_j): at least one component must explain each input
- L_var = -Σ log Var(d_j): prevents dead components  
- L_tc = ||Corr(d) - I||²_F: prevents redundant components

This is the neural analogue of the log-determinant in GMMs, same as Paper 2.

### Calibration (NegLogSoftmin)

```
y_j = NegLogSoftmin(d)_j = -log(exp(-d_j) / Σ_k exp(-d_k)) = d_j + log Z
```

Where Z = Σ_k exp(-d_k) is the partition function.

Properties:
- Distance in, distance out. Convention preserved.
- Adds a scalar (log Z) that is the same for all components. Relative distances unchanged.
- exp(-y_j) = r_j (the responsibility). Outputs are probabilistically calibrated.
- The shift absorbs the partition function so downstream layers receive properly normalized representations.
- Differentiable. Gradients flow through it from the supervised loss.

### Interface

Forward pass returns:
- y: calibrated distances (for next layer)
- d: raw distances (for volume control auxiliary loss)

---

## The Supervised Model

### Architecture

```
x → [ImplicitEM Layer] → y → Linear → LayerNorm → CrossEntropy(·, labels)
                          ↑
                     aux_loss(d)
```

Expanded:
```
d = Softplus(W₁x + b₁)                          # raw distances
y = d + log Σ exp(-d)                             # NegLogSoftmin (calibrated distances)
h = LayerNorm(W₂y + b₂)                          # project to class space
total_loss = CE(h, labels) + λ · aux_loss(d)      # supervised + volume control
```

### Why LayerNorm

- y values can be positive or negative (d > 0 but log Z can be negative when all distances are large)
- LayerNorm normalizes per-sample across features: zero mean, unit variance
- Prevents scale of ImplicitEM layer output from coupling with classifier learning dynamics
- Standard practice; not a theoretical novelty

### Why Not BatchNorm

BatchNorm normalizes per-feature across the batch. This would interfere with the per-sample probabilistic calibration that NegLogSoftmin provides. LayerNorm preserves the within-sample structure.

---

## Experimental Plan

### Experiment 1: Single-Layer CE Analysis

**Goal:** Show that single-layer CE already has implicit volume control from labels.

**Method:** Train single linear layer on MNIST with CE. Measure:
- Dead units: 0 expected (labels force all classes to receive gradient)
- Redundancy: Low expected (classes must separate)
- Weight visualization: Digit prototypes expected

**Point:** Labels provide what InfoMax provides in the unsupervised case. Brief demonstration, not the main result.

### Experiment 2: Theorem Verification (Supervised Variant)

**Goal:** Verify the clamped gradient-responsibility identity.

**Method:** Random logits, random labels. Verify:
```
∂L_CE/∂z_j = softmax(z)_j - 1[j=y]
```
to floating-point precision.

Same style as Paper 2 Experiment 1. Plot gradient vs (softmax - indicator) on y=x.

### Experiment 3: Two-Layer Ablation (Main Result)

**Goal:** Show that volume control is necessary at intermediate layers even with supervised training.

**Configurations:**

| Config | ImplicitEM Layer | Prediction |
|--------|-----------------|------------|
| CE only | No ImplicitEM layer. Two linear layers, Softplus, LayerNorm. | Baseline. Intermediate representation unstructured. |
| CE + NegLogSoftmin only | Softplus + NegLogSoftmin, no aux loss | Calibrated but unconstrained. Possible degeneration. |
| CE + LSE only | Softplus + NegLogSoftmin + LSE aux loss | EM dynamics but no volume control. Intermediate collapse. |
| CE + LSE + var | + variance penalty | Alive but redundant intermediates. |
| CE + LSE + var + tc | Full ImplicitEM layer | Structured mixture at intermediate layer. |
| CE + var + tc (no LSE) | Variance + decorrelation but no LSE | Whitened intermediates. Different dynamics. |

**Metrics (at intermediate layer):**
- Dead units (variance < threshold)
- Redundancy (||Corr - I||²_F)
- Responsibility entropy
- Weight visualization (W₁ reshaped to 28×28)

**Metrics (at output):**
- Classification accuracy
- Linear probe on intermediate activations

**Key prediction:** The ablation pattern from Paper 2 replicates at the intermediate layer, even with supervised gradients flowing through. Volume control is needed independently of supervision.

### Experiment 4: Intermediate Feature Visualization

**Goal:** Show that ImplicitEM layer learns interpretable prototypes.

**Method:** Visualize W₁ rows as 28×28 images for each ablation config.

**Prediction:**
- Full ImplicitEM: Digit prototypes (like Paper 2 unsupervised model)
- CE only: Unstructured weights (like SAE encoder in Paper 2)
- The supervised gradient alone doesn't produce structured intermediate representations

### Experiment 5: Comparison with Unsupervised Model

**Goal:** Compare intermediate representations of supervised ImplicitEM model with the unsupervised model from Paper 2.

**Method:** Same hidden dimension (64). Compare:
- Weight visualizations
- Responsibility distributions
- Sparsity patterns
- Linear probe accuracy

**Prediction:** Similar structure. Supervised model may have cleaner class-aligned prototypes due to gradient signal from labels. Unsupervised model captures sub-digit variation.

### Experiment 6: Training Dynamics

**Goal:** Check whether the optimization anomalies from Paper 2 appear in the supervised case.

**Method:** SGD vs Adam across learning rates. Track:
- Loss trajectories
- Intermediate layer metrics over training
- Classification accuracy

**Questions:**
- Does SGD learning-rate insensitivity appear for the auxiliary loss?
- Does Adam achieve lower auxiliary loss without better intermediate features?
- Does the supervised gradient disrupt or preserve the EM dynamics?

---

## Open Questions

### Does the supervised gradient interfere with intermediate EM dynamics?

The intermediate layer receives two gradient signals:
1. From the auxiliary loss (pure EM dynamics, same as unsupervised)
2. From the classification loss (supervised, flowing back through W₂ and LayerNorm)

These could cooperate or compete. If the supervised gradient wants a representation that's good for classification but geometrically degenerate, it fights the volume control. If it wants a representation that's structured and diverse, it cooperates.

Prediction: They mostly cooperate, because a well-formed mixture representation is naturally good for classification. But the balance may depend on λ (weight of auxiliary loss).

### What is the right λ?

The relative weight of auxiliary loss vs supervised loss. Too low: volume control ineffective. Too high: supervised signal overwhelmed.

This might interact with the finding from Paper 2 that the objective has a null space. If auxiliary loss and supervised loss are somewhat orthogonal, λ may not matter much.

### Does this scale beyond MNIST?

Same limitation as Paper 2. The theory holds for any differentiable parameterization. Whether the competitive dynamics work with hundreds of components on natural images is untested.

### Multiple ImplicitEM layers?

The theory prescribes volume control at every layer. Can you stack ImplicitEM layers? Each one performs its own EM over its inputs. This is the "hierarchical mixtures" idea from Paper 2's future directions. Not in scope for this paper, but the architecture naturally extends.

---

## Paper Structure (Draft)

1. **Introduction:** Paper 1 identified three regimes. Paper 2 validated unsupervised. This paper validates supervised. The key question: what provides volume control in supervised learning?

2. **Theory:** Brief recap of implicit EM. Single-layer CE analysis showing labels provide output-layer volume control. The gap: intermediate layers lack volume control even in supervised networks.

3. **The ImplicitEM Layer:** Derivation from theory. Distance computation + volume control + calibration. Each component traced to theoretical requirement.

4. **Experimental Validation:** Theorem verification. Two-layer ablation. Feature visualization. Comparison with unsupervised. Training dynamics.

5. **Discussion:** Volume control is needed at every layer. Labels only control the output. Reframes BatchNorm/LayerNorm/weight decay as implicit volume control. Connection to deep learning stability heuristics.

6. **Conclusion:** Implicit EM theory prescribes supervised architectures. Volume control is the missing piece for intermediate representations.

---

## Relation to Paper 2's Future Directions

Paper 2 listed these as future work:
- ✅ **Supervised regularization:** "Adding InfoMax during supervised learning would encourage mixture-like representations even with labels."
- ✅ **Layer-wise pretraining:** "Each layer learns a mixture model over its inputs. Hierarchical mixtures, all the way up." (We do this end-to-end, not greedily.)
- **Conditioning pretrained models:** Not in scope.
- **Activation geometry:** Softplus vs ReLU. We use Softplus; could compare.
- **Attention:** Not in scope for this paper.

---

## Implementation Notes

- NegLogSoftmin: `y = d + torch.logsumexp(-d, dim=1, keepdim=True)`
- Volume control: Reuse `combined_loss(d, W, config)` from Paper 2 code unchanged
- Softplus instead of ReLU for intermediate layer
- LayerNorm before CE loss
- GPU-cached MNIST data loaders already implemented
- Geometric metrics from `geometric_metrics.py` applicable to intermediate representations