# Supervised Model: Architecture and Theoretical Justification

## Overview

We test a two-layer supervised model on MNIST. Every component is derived from implicit EM theory. Nothing is added for empirical convenience. Nothing the theory requires is omitted.

The model has two stages:

1. **ImplicitEM layer:** Learns a mixture model over inputs. Produces calibrated distance representations with volume control.
2. **Classification head:** Projects calibrated distances to class logits. Standard cross-entropy.

```
x → Linear → Softplus → [volume control loss] → NegLogSoftmin → Linear → LayerNorm → CE(·, y)
    \_________________  _____________________/                    \____________  ____________/
            ImplicitEM layer                                        Classification head
```

## Component-by-Component Justification

### Layer 1: Linear (W₁x + b₁)

**Theory source:** Oursland [2024]. A linear layer computes projections onto learned directions. Under the distance interpretation, each row of W₁ defines a component. The pre-activation z_j = w_j⊤x + b_j measures signed deviation from a learned reference surface.

**Dimensions:** W₁ ∈ ℝ^{K×784} where K is the number of mixture components (e.g., 64). Each of the K rows defines one component of the mixture model.

**What it learns:** Prototype directions. Each row of W₁ converges to a direction in input space that defines one mixture component's territory.

### Activation: Softplus

**Theory source:** Distance interpretation requires non-negative outputs. Distances cannot be negative.

**Why not ReLU:** ReLU maps the entire half-space where z < 0 to distance zero. This creates a degenerate region where all inputs are "perfectly matched" to a component. The prototype is not a point but a half-space. For mixture modeling, this is geometrically pathological — a component claims an infinite region at zero cost.

**Why Softplus:** Softplus(z) = log(1 + exp(z)) is always positive, smooth, and has no flat region. Every input has a unique, nonzero distance to every component. The prototype surface (minimum distance) is a smooth manifold rather than a hard boundary.

**What it computes:** d_j = Softplus(w_j⊤x + b_j) is the distance from input x to component j. Lower values indicate better matches.

### Volume Control: LSE + Variance + Decorrelation (Auxiliary Loss)

**Theory source:** Oursland [2025], validated in Oursland [2026]. Neural LSE objectives lack the log-determinant term that prevents collapse in Gaussian mixture models. Without volume control, representations degenerate.

Applied to raw distances d as an auxiliary loss. Three terms:

**LSE loss:** L_LSE = -log Σ exp(-d_j)

At least one component must explain each input. This is the negative log marginal likelihood. It provides the EM dynamics — gradients equal responsibilities.

**Variance penalty:** L_var = -Σ log Var(d_j)

Each component must maintain non-zero variance across the batch. Prevents dead components. Corresponds to the diagonal of the log-determinant in GMMs.

**Decorrelation penalty:** L_tc = ||Corr(d) - I||²_F

Components must respond differently across inputs. Prevents redundant components. Corresponds to the off-diagonal of the log-determinant in GMMs.

**Why the intermediate layer needs this:** Labels provide volume control at the output layer — every class must win for its examples, so no class can die or become redundant. But labels do not control intermediate representations. The supervised gradient flowing back from the classification head only demands "produce something classifiable." It is indifferent to the geometric health of intermediate distances. Without explicit volume control, intermediate components can collapse, become redundant, or degenerate — even in a fully supervised network.

### Calibration: NegLogSoftmin

**Theory source:** Type-preserving calibration derived from probability chain. See neg_log_softmin.md.

**Computation:** y_j = d_j + log Σ exp(-d_k)

Adds a scalar (log of partition function) that is identical across components within each sample. Relative distances unchanged. Ranking unchanged.

**Purpose:** After this layer, exp(-y_j) = r_j (the responsibility). Distances carry proper probabilistic semantics. The downstream classification head receives a representation where absolute values have fixed meaning, regardless of the scale of the upstream distances.

**Why it's needed:** Without calibration, the classification head must learn to compensate for whatever arbitrary scale the ImplicitEM layer produces. With calibration, the representation is standardized. The partition function is absorbed. The classification head operates on probabilistically meaningful quantities.

### Layer 2: Linear (W₂y + b₂)

**Theory source:** Standard. Projects K-dimensional calibrated distances to C class logits (C = 10 for MNIST).

**Dimensions:** W₂ ∈ ℝ^{10×K}. Maps from mixture component space to class space.

**What it learns:** The assignment from mixture components to classes. Which components correspond to which digit. This is a linear readout of the mixture representation.

### LayerNorm

**Theory source:** Practical, not theoretical. Prevents scale coupling between ImplicitEM layer and classification head.

**Why LayerNorm over BatchNorm:** LayerNorm normalizes per-sample across features. BatchNorm normalizes per-feature across the batch. LayerNorm preserves the within-sample probabilistic structure that NegLogSoftmin provides. BatchNorm would destroy it.

**Input characteristics:** NegLogSoftmin outputs can be positive or negative. d_j > 0 always (Softplus), but log Z can be negative when all distances are large (all components far from the input, Z < 1). LayerNorm handles mixed-sign inputs correctly.

### Loss: Cross-Entropy

**Theory source:** Oursland [2025] Section 4.3. Cross-entropy has LSE structure with a label clamp. The gradient is:

∂L/∂z_j = softmax(z)_j - 𝟙[j = y]

This is implicit EM with the correct class clamped to responsibility 1. Competition among incorrect classes remains responsibility-weighted.

**Applied to:** The output of LayerNorm. Standard supervised classification.

### Total Loss

```
total_loss = CE(h, labels) + λ · (L_LSE + λ_var · L_var + λ_tc · L_tc)
```

Two terms:
- **CE:** Supervised signal. Makes the model classify correctly.
- **Auxiliary (volume control):** Structural signal. Makes the intermediate representation a well-formed mixture.

These are not in conflict. A well-formed mixture representation is naturally good for classification. The auxiliary loss does not fight supervision — it supplements the volume control that supervision provides at the output but does not provide at intermediate layers.

## What This Model Is

Three equivalent descriptions:

**A theory-prescribed supervised network.** Every component traces to a requirement of implicit EM theory. The ImplicitEM layer is derived, not designed. The classification head is standard. The auxiliary loss is the neural log-determinant.

**A supervised hierarchical mixture model.** Layer 1 learns a mixture model over inputs (unsupervised EM dynamics + volume control). Layer 2 assigns mixture components to classes (supervised EM dynamics, label-clamped). Two levels of implicit EM, composed.

**A minimal test case.** The model exists to test the theory. Can implicit EM prescribe supervised architectures? Does volume control matter at intermediate layers? The model is the simplest architecture that tests these questions.

## What This Model Is Not

**Not a novel architecture for MNIST classification.** We are not trying to beat state-of-the-art. We are testing whether the theory's prescriptions produce a working model with predictable failure modes.

**Not a deep network.** Two layers. One ImplicitEM layer, one classification head. Depth is not the point. Volume control at intermediate representations is the point.

**Not optimized.** No learning rate schedules, no data augmentation, no architectural search. Hyperparameters set to simple defaults (λ_var = λ_tc = 1.0). Same methodology as Paper 2: build what the theory says, test whether it works.

## Parameter Count

| Component | Parameters |
|-----------|-----------|
| W₁ (K × 784) + b₁ (K) | 785K |
| W₂ (10 × K) + b₂ (10) | 10K + 10 |
| LayerNorm (scale + bias) | 20 |
| **Total** | **795K + 30** |

For K = 64: 50,910 parameters.

No decoder. No reconstruction pathway. The only learned components are the mixture layer, the classifier, and LayerNorm's affine parameters.

## Ablation Design

The theory makes specific predictions about what each component contributes. Removing components should produce specific, predictable failures.

| Config | Components | Prediction |
|--------|-----------|------------|
| Baseline MLP | Linear + Softplus + Linear + LayerNorm + CE | Classification works. Intermediate representation unstructured. No EM dynamics at layer 1. |
| + NegLogSoftmin | Add calibration, no aux loss | Calibrated but unconstrained. Possible degeneration of intermediates. |
| + LSE only | Add LSE aux loss | EM dynamics at layer 1, but no volume control. Intermediate collapse. |
| + LSE + var | Add variance penalty | Components alive but redundant. |
| + LSE + var + tc | Full ImplicitEM layer | Structured mixture at intermediate layer. All components alive, diverse, competitive. |
| + var + tc (no LSE) | Volume control without EM | Whitened intermediates. No mixture structure. Lower responsibility entropy. |

The ablation tests volume control inside a supervised network. The prediction is that the same pattern from Paper 2 (unsupervised) replicates at the intermediate layer — because from the volume control perspective, the intermediate layer is unsupervised regardless of what happens at the output.

## Metrics

### Intermediate Layer Health (applied to distances d)
- Dead units: components with Var(d_j) < threshold
- Redundancy: ||Corr(d) - I||²_F
- Responsibility entropy: H(softmin(d)) averaged over samples
- Weight visualization: rows of W₁ reshaped to 28×28

### Output Quality
- Classification accuracy on MNIST test set
- Linear probe accuracy on intermediate activations (freeze ImplicitEM layer, train linear classifier)

### Geometric Metrics (from geometric_metrics.py)
- Weight-input alignment
- Template correlation
- Activation-cosine correlation
- Center-surround structure
- Boundary placement

These distinguish whether the ImplicitEM layer learns prototypes (templates) or boundary detectors (Mahalanobis normals).