# Encoder Study: Experimental Validation Plan

## Overview

This document outlines experiments to validate the decoder-free sparse autoencoder derived from implicit EM theory. The experiments must accomplish three goals:

1. **Prove the mechanism:** Implicit responsibilities and competitive specialization behave as predicted
2. **Prove the object:** We learn a "Gaussian Component Field," not a traditional GMM
3. **Prove utility:** The representations are useful for downstream tasks and interpretation

The theoretical stack being validated:

- **Layer 1 (Oursland 2024):** Linear layers compute distances to prototypes
- **Layer 2 (Oursland 2025):** LSE objectives yield ∂L/∂d_j = -r_j (implicit EM)
- **Layer 3 (This work):** LSE + InfoMax = decoder-free sparse autoencoder

## Experiment Priority Tiers

### Tier 1: Must-Have (Core Validation)

These experiments validate the theoretical claims and are required for the paper.

### Tier 2: Should-Have (Mechanistic Insights)

These experiments provide deeper understanding and strengthen the contribution.

### Tier 3: Nice-to-Have (Extensions)

These experiments broaden applicability but are not essential.

---

## Tier 1 Experiments

### 1.1 Verify Gradient = Responsibility Identity

**Goal:** Prove the core theorem is implemented correctly.

**Method:**
- During a single training step, record:
  - Calculated responsibility: r_j = softmax(-E_j)
  - Actual gradient: ∂L_LSE/∂E_j
- Create scatter plot of r_j vs ∂L_LSE/∂E_j

**Expected Result:** Perfect identity line (y = x). This is the "mic drop" that proves implicit EM is real.

**Visualization:** Scatter plot with identity line overlay.

---

### 1.2 Ablation: Collapse and Volume Control

**Goal:** Demonstrate that InfoMax prevents collapse as predicted by theory.

**Method:** Train five configurations on same data:

| Configuration | LSE | Variance | Correlation | Weight Reg |
|---------------|-----|----------|-------------|------------|
| LSE only | ✓ | | | |
| LSE + Variance | ✓ | ✓ | | |
| LSE + Variance + Corr | ✓ | ✓ | ✓ | |
| LSE + Full InfoMax | ✓ | ✓ | ✓ | ✓ |
| InfoMax only (no LSE) | | ✓ | ✓ | ✓ |

**Metrics:**
- Dead unit count (variance < threshold)
- Global usage entropy: H(E_x[r(x)])
- Redundancy score: ||off-diag(Corr(A))||
- Weight redundancy: ||W^T W - I||²
- Feature similarity matrix visualization

**Expected Results:**
- LSE only → collapse (one component dominates)
- Variance only → no dead units but redundant features
- Full InfoMax → stable, diverse, no collapse

**Visualization:** 2x3 grid showing feature similarity matrices for each configuration.

---

### 1.3 Responsibility Dynamics Over Training

**Goal:** Show that responsibilities behave like EM: competition sharpens, components specialize.

**Method:** Track throughout training:
- Per-sample responsibility entropy: H(r(x))
- Global usage distribution: E_x[r(x)]
- Effective number of components: N_eff = exp(H(E_x[r(x)]))

**Expected Results:**
- Per-sample entropy decreases (competition sharpens)
- Global usage stays distributed (no single winner)
- Two-timescale behavior possible: responsibilities stabilize before weights

**Visualization:** 
- Line plots of entropy metrics over training epochs
- Heatmap of component usage across data clusters at different training stages

---

### 1.4 Implicit Decoder Validation

**Goal:** Prove "decoder-free" doesn't mean "information-free."

**Method:**
- Train encoder with LSE + InfoMax (no reconstruction loss)
- Periodically compute reconstruction using W^T as decoder: x̂ = W^T a
- Compare to tied-weight autoencoder trained with explicit reconstruction

**Metrics:**
- Reconstruction MSE using frozen W^T
- Compare to baseline autoencoder MSE

**Expected Result:** Reconstruction error using W^T decreases during training and converges to baseline, even though reconstruction was never optimized.

**Why this matters:** Proves the encoder learns a reversible basis through implicit EM alone.

---

### 1.5 Standard SAE Benchmark

**Goal:** Practical validation against existing methods.

**Dataset:** Transformer activations (GPT-2 small MLP outputs or residual stream)

**Baselines:**
- Vanilla SAE (encoder + decoder, L1 sparsity)
- TopK SAE
- Gated SAE

**Metrics:**
- Reconstruction loss (using W^T for ours)
- Sparsity (L0 norm, responsibility entropy)
- Parameter count / memory footprint
- Training throughput (samples/second)
- Linear probe accuracy on downstream task

**Expected Result:** Comparable quality with ~50% fewer parameters (no decoder storage).

---

## Tier 2 Experiments

### 2.1 Synthetic Geometry: Field vs GMM

**Goal:** Prove we learn a "Component Field," not a traditional GMM.

**Datasets (2D/3D for visualization):**
1. True GMM with elliptical clusters (easy for GMM)
2. Curved manifold (two moons, Swiss roll) where GMM struggles
3. Union of subspaces (mixture of lines/planes)

**Baselines:**
- GMM (full covariance)
- K-means
- ICA

**Visualizations:**
- Learned weight vectors as arrows in input space
- Responsibility partition as soft tessellation
- Coverage heatmap across space

**Metrics:**
- Geometric coverage: nearest-component energy over space
- Components needed to cover manifold at fixed error
- Alignment with true structure (if known)

**Expected Results:**
- On true GMM: model recovers components (or produces equivalent field)
- On curved manifold: field tiles locally, GMM fails
- On union of subspaces: field captures local directions

---

### 2.2 Component Field Characterization

**Goal:** Show features are scattered local directions, not orthogonal global components.

**Method:**
- Analyze pairwise angles between weight vectors
- Compare to PCA/ICA (which force orthogonality)
- Visualize "tangled" structure of weight vectors

**Metrics:**
- Distribution of pairwise cosine similarities
- Effective dimensionality of W
- Comparison to orthogonal baselines

**Expected Result:** Weight vectors are NOT organized into orthogonal sets; they form a scattered field that adapts to data geometry.

---

### 2.3 Batch Size Effects

**Goal:** Understand how batch size affects learning dynamics.

**Method:** Train with batch sizes [32, 128, 512, 2048, full batch]

**Metrics:**
- Convergence speed
- Final responsibility distribution
- Stability of training

**Expected Result:** Larger batches give cleaner responsibility estimates. May mirror mini-batch EM literature.

---

### 2.4 Feature Co-occurrence Analysis

**Goal:** Distinguish from clustering (winner-take-all) vs component field (winner-take-most).

**Method:** Analyze distribution of active features per input.

**Visualization:** Histogram of L0 norm (number of features with responsibility > threshold)

**Expected Results:**
- Clustering: ~1 feature active per input
- Component Field: sparse set (5-10) features active per input
- Distribution should be Poisson-like, not binary

---

### 2.5 Specialization and Interpretability

**Goal:** Show learned features are coherent and interpretable.

**Method:**
- For each unit: collect top-k activating samples
- For vision: show image patches, spatial activation maps
- For language: show top contexts/tokens

**Visualization:** Gallery of feature visualizations with top-activating examples.

**Expected Result:** Each feature captures semantically coherent concept or local direction.

---

## Tier 3 Experiments

### 3.1 Out-of-Distribution Behavior

**Goal:** Test whether field structure provides better OOD detection.

**Method:**
- Train on MNIST, test on FashionMNIST
- Train on CIFAR-10, test on SVHN

**Metrics:**
- Max responsibility (confidence proxy)
- Minimum energy across components
- Responsibility entropy H(r(x))

**Expected Result:** Higher entropy / higher min-energy on OOD data (uncertain rather than confidently wrong).

---

### 3.2 Mahalanobis Alignment Check

**Goal:** Connect to Paper 1 (Mahalanobis interpretation).

**Method:** On synthetic Gaussian clusters with known covariance:
- Check if learned W rows align with principal directions
- Test if weight regularization improves alignment

**Metrics:**
- Cosine similarity between learned vectors and true eigenvectors
- Effect of λ_wr on alignment

---

### 3.3 Different Domains

**Goal:** Show generality beyond primary dataset.

**Options:**
- Vision: CIFAR-10, ImageNet patches
- Language: Different model sizes, different layers
- Audio: Speech features

---

### 3.4 Computational Scaling

**Goal:** Characterize efficiency advantages.

**Metrics:**
- Parameters vs quality curves
- Training throughput vs baseline SAEs
- Memory footprint comparison
- Scaling to very wide feature spaces

---

## Experimental Infrastructure

### Datasets

**Primary:**
- MNIST/FashionMNIST (simple, fast iteration)
- GPT-2 small activations (relevant to SAE community)

**Secondary:**
- 2D/3D synthetic (for visualization)
- CIFAR-10 (vision benchmark)

### Baselines

Essential:
- GMM (full covariance)
- K-means
- Vanilla SAE (encoder + decoder + L1)

Recommended:
- TopK SAE
- Barlow Twins / VICReg (variance + covariance regularization)
- ICA

Optional:
- Gated SAE
- VQ-VAE

### Metrics Summary

| Metric | What it validates |
|--------|-------------------|
| Gradient = Responsibility scatter | Core theorem |
| Dead unit count | Anti-collapse |
| Feature similarity matrix | Redundancy prevention |
| Responsibility entropy over training | EM dynamics |
| Reconstruction via W^T | Information preservation |
| Linear probe accuracy | Downstream utility |
| L0 distribution | Field vs cluster |
| Geometric coverage | Manifold tiling |

### Key Visualizations

1. **Scatter: Gradient vs Responsibility** — Proves the math
2. **Grid: Feature similarity matrices by configuration** — Proves ablation
3. **Line: Responsibility entropy over training** — Proves EM dynamics
4. **Gallery: Top-activating samples per feature** — Proves interpretability
5. **2D/3D: Weight vectors as arrows** — Proves field structure
6. **Histogram: Features active per sample** — Proves not clustering

---

## Recommended Paper Flow

1. **Introduction:** Decoder-free SAEs from first principles
2. **Theory:** Derivation (LSE + InfoMax)
3. **Experiment 1.1:** Gradient = Responsibility (validates theorem)
4. **Experiment 1.2:** Ablation (validates each term)
5. **Experiment 1.3:** EM dynamics (validates mechanism)
6. **Experiment 2.1:** Synthetic geometry (validates "field" concept)
7. **Experiment 1.5:** Benchmark (validates practical utility)
8. **Experiment 2.5:** Interpretability (validates usefulness)
9. **Discussion:** When/why to use this architecture

---

## Minimum Viable Experiment Set

If constrained, run these four:

1. **Gradient = Responsibility scatter** (proves theorem)
2. **Ablation grid with similarity matrices** (proves volume control)
3. **Synthetic 2D showing field vs GMM** (proves concept)
4. **One real dataset with linear probe + interpretability gallery** (proves utility)

This validates all three layers of the theoretical stack with minimal experiments.

---

## Open Questions for Experiments

1. **What's the right responsibility threshold for "active"?**
2. **How do we fairly compare parameter counts (ours has no decoder)?**
3. **Is reconstruction via W^T the right measure, or should we use something else?**
4. **What batch size is needed for stable responsibility estimates?**
5. **How does the choice of activation (ReLU vs others) affect results?**