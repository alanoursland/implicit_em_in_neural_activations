# Deriving Decoder-Free Sparse Autoencoders from First Principles

## Paper Outline

### Abstract
- Sparse autoencoders learn interpretable features but require decoders
- We derive decoder-free SAEs from implicit EM theory
- Key insight: LSE objectives give EM structure, but collapse without volume control
- InfoMax regularization serves as volume control (analogous to log-determinant in GMMs)
- Experiments confirm: principled objective matches standard SAEs, no decoder needed

---

### 1. Introduction

**1.1 Opening: The Decoder Question**
- Sparse autoencoders are useful for interpretability
- Standard architecture: encoder + decoder + sparsity penalty
- The decoder ensures information preservation via reconstruction
- Question: Is reconstruction the only way to preserve information?

**1.2 This Paper**
- We derive decoder-free SAEs from first principles
- The derivation follows from implicit EM theory
- Log-sum-exp objectives perform EM implicitly (gradient = responsibility)
- But LSE alone collapses—missing volume control
- InfoMax regularization provides the missing piece
- Result: principled unsupervised objective, no decoder

**1.3 Contribution Statement**
- The derivation is the contribution, not the architecture
- We explain *why* decoder-free SAEs work
- Three experiments validate the theoretical claims

**1.4 Roadmap**
- Section 2: Background on implicit EM
- Section 3: The collapse problem
- Section 4: InfoMax as volume control
- Section 5: Experiments
- Section 6: Related work
- Section 7: Discussion

---

### 2. Background: Implicit EM in Gradient Descent

**2.1 The Log-Sum-Exp Identity**
- Define component energies E_j(x), lower = better explanation
- Define objective: L = -log Σ_j exp(-E_j)
- State the identity: ∂L/∂E_j = r_j where r_j = softmax(-E)_j
- This is exact, not approximate

**2.2 What This Means**
- The gradient with respect to each energy IS its responsibility
- No auxiliary E-step computation needed
- EM happens implicitly in backpropagation
- M-step happens in the optimizer update

**2.3 Supervised vs Unsupervised**
- Cross-entropy has LSE structure (supervised)
- For unsupervised: LSE marginal says "someone must explain each input"
- Natural choice, but incomplete

---

### 3. The Problem: Collapse Without Volume Control

**3.1 The Collapse Failure Mode**
- Pure LSE allows one component to claim all inputs
- That component's responsibility → 1
- Other components receive vanishing gradient
- They die; representation degenerates

**3.2 The Mixture Model Analogy**
- Same problem exists in Gaussian mixture models
- GMMs prevent collapse via log-determinant term
- log det(Σ) penalizes small covariance
- A component cannot shrink to a point

**3.3 The Gap in Neural Networks**
- Neural energy models have no log-determinant equivalent
- Implicit EM machinery exists (the LSE identity)
- Volume control is missing
- This is what we must supply

---

### 4. The Solution: InfoMax as Volume Control

**4.1 Design Goals**
- Prevent collapse: every component must be active
- Prevent redundancy: components must be distinct
- Inspired by InfoMax principles (Linsker, Bell & Sejnowski)
- No claim of computing mutual information exactly

**4.2 Anti-Collapse: Variance Penalty**
- Dead component has zero variance across dataset
- Penalty: L_var = -Σ_j log Var(A_j)
- Diverges if any component collapses
- Forces every component to respond to some inputs

**4.3 Anti-Redundancy: Decorrelation Penalty**
- Redundant components are correlated
- Penalty: L_tc = ||Corr(A) - I||²
- Forces components to encode different structure
- Approximates independence at second-order statistics

**4.4 The Complete Objective**
- L = L_LSE + λ_var · L_var + λ_tc · L_tc
- Expanded: -log Σ exp(-E_j) - λ_var Σ log Var(A_j) + λ_tc ||Corr(A) - I||²
- Optional: weight redundancy term λ_wr ||W^T W - I||²

**4.5 Intuition: Attraction vs Structure**
- LSE is attractive: pulls prototypes toward data (maximize likelihood)
- InfoMax is structural: forces selectivity and diversity
- Equilibrium: competitive coverage of the data manifold
- Like differentiable vector quantization with soft assignments

**4.6 Architecture**
- z = Wx + b (linear layer computes distances)
- E = φ(z) (activation produces energies; ReLU, softplus, etc.)
- Responsibilities implicit: r = softmax(-E)
- No decoder, no reconstruction

---

### 5. Experiments

**5.1 Experiment 1: Verifying the Identity**
- Goal: Confirm ∂L_LSE/∂E_j = r_j
- Method: Single forward/backward pass, record gradient and responsibility
- Output: Scatter plot, should be identity line
- Result: [Show Figure 1]

**5.2 Experiment 2: Ablation**
- Goal: Show each term is necessary
- Configurations:
  - A: LSE only
  - B: LSE + variance
  - C: LSE + variance + decorrelation
  - D: Variance + decorrelation only (no LSE)
- Metrics: Dead units, redundancy score, feature usage
- Expected results:
  - A: Collapse (one component dominates)
  - B: No dead units, but redundancy
  - C: Stable, diverse features
  - D: No EM structure, different behavior
- Output: Table + Figure 2

**5.3 Experiment 3: Benchmark**
- Goal: Show competitive with standard SAE
- Dataset: MNIST (or LLM activations)
- Baseline: Standard SAE (encoder + decoder + L1)
- Metrics:
  - Reconstruction MSE (ours uses W^T as implicit decoder)
  - Sparsity (L0 norm)
  - Parameter count
- Expected: Comparable quality, ~50% fewer parameters
- Output: Table (Figure 3)

---

### 6. Related Work

**6.1 Energy-Based Models**
- LeCun et al. 2006
- Our LSE is negative log marginal under mixture of energies
- InfoMax terms regularize the energy landscape

**6.2 Sparse Coding**
- Olshausen & Field 1996
- Standard: reconstruction + L1 sparsity
- Ours: likelihood + volume control, no decoder

**6.3 InfoMax and ICA**
- Bell & Sejnowski 1995
- Variance ≈ entropy proxy, decorrelation ≈ independence proxy
- We borrow operational definitions, not theoretical claims

**6.4 Self-Supervised Learning**
- VICReg, Barlow Twins use similar variance/covariance terms
- Different context (contrastive learning), similar regularization insight

**6.5 Implicit EM**
- Oursland 2025 establishes the gradient = responsibility identity
- We build on this to derive the complete objective

---

### 7. Discussion

**7.1 What We Showed**
- Decoder-free SAEs arise from implicit EM + volume control
- The derivation is principled, not heuristic
- Each term has a clear role

**7.2 What We Didn't Show**
- That this is *better* than existing SAEs
- We showed competitive, not superior
- Large-scale validation left to future work

**7.3 Limitations**
- Single-layer experiments
- Limited to MNIST / small scale
- Theoretical claims stronger than empirical validation
- Did not explore all activation functions

**7.4 Future Work**
- Scale to LLM interpretability applications
- Multi-layer extensions
- Tighter theoretical analysis of convergence
- Connection to other implicit inference methods

---

### 8. Conclusion

- Sparse autoencoders traditionally require decoders for information preservation
- We derived a decoder-free objective from implicit EM theory
- Log-sum-exp provides EM structure; InfoMax provides volume control
- Together they replace reconstruction with competitive coverage
- The decoder was compensating for missing volume control
- With proper regularization, implicit EM suffices

---

## Figures

**Figure 1: Theorem Verification**
- Scatter plot: x = r_j, y = ∂L/∂E_j
- Should show perfect y = x line
- Caption: "The implicit EM identity holds exactly: gradients equal responsibilities."

**Figure 2: Ablation Results**
- Option A: 2×2 grid of feature correlation matrices by configuration
- Option B: Bar chart showing dead units / redundancy by configuration
- Caption: "Each regularization term serves a distinct purpose. LSE alone collapses; InfoMax provides necessary structure."

**Figure 3: Benchmark Comparison**
- Table comparing: Model, Recon. MSE, Sparsity (L0), Parameters
- Or bar chart of same
- Caption: "Decoder-free objective matches standard SAE with fewer parameters."

---

## Key Equations

1. **LSE Objective:** $L_{\text{LSE}} = -\log \sum_j \exp(-E_j(x))$

2. **The Identity:** $\frac{\partial L_{\text{LSE}}}{\partial E_j} = r_j = \frac{\exp(-E_j)}{\sum_k \exp(-E_k)}$

3. **Variance Penalty:** $L_{\text{var}} = -\sum_j \log \text{Var}(A_j)$

4. **Decorrelation Penalty:** $L_{\text{tc}} = \|\text{Corr}(A) - I\|^2$

5. **Complete Objective:** $L = L_{\text{LSE}} + \lambda_{\text{var}} L_{\text{var}} + \lambda_{\text{tc}} L_{\text{tc}}$

6. **Architecture:** $z = Wx + b, \quad E = \phi(z)$

---

## Target Length by Section

| Section | Pages |
|---------|-------|
| Abstract | 0.15 |
| 1. Introduction | 0.75 |
| 2. Background | 1.00 |
| 3. Collapse Problem | 0.75 |
| 4. Solution | 1.50 |
| 5. Experiments | 1.50 |
| 6. Related Work | 0.50 |
| 7. Discussion | 0.50 |
| 8. Conclusion | 0.25 |
| References | 0.50 |
| **Total** | **7.40** |