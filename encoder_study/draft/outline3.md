# Deriving Decoder-Free Sparse Autoencoders from First Principles

## Paper Outline (Revised)

### Abstract

Sparse autoencoders learn interpretable features but impose a "decoder tax"—doubling parameters and compute for reconstruction. We show the decoder is unnecessary. By viewing neural activations as energies, we derive a decoder-free objective from implicit expectation-maximization theory. The key insight: log-sum-exp objectives perform EM implicitly (gradient equals responsibility), but collapse without volume control. We identify InfoMax regularization—maximizing variance, minimizing correlation—as the missing piece, analogous to the log-determinant in Gaussian mixture models. The resulting objective enables a single encoder to learn sparse, diverse features without reconstruction. Experiments confirm the theoretical predictions: LSE alone collapses; InfoMax provides necessary structure; the combination matches standard SAEs with half the parameters.

---

### 1. Introduction

**1.1 The Decoder Tax**
- Sparse autoencoders extract interpretable features from neural networks
- Standard architecture: encoder + decoder + L1 sparsity penalty
- The decoder doubles parameters and compute
- Reconstruction ensures information preservation—but is it fundamental or compensatory?

**1.2 The Central Question**
- Encoder-only SAEs exist and work empirically
- But they lack principled derivation
- What is the decoder actually *doing* for the encoder?

**1.3 This Paper**
- We derive decoder-free SAEs from first principles via implicit EM
- Log-sum-exp objectives perform EM implicitly (gradient = responsibility)
- But LSE alone collapses—missing volume control
- InfoMax regularization provides the missing piece
- Result: principled encoder-only objective, no reconstruction needed

**1.4 Contribution Statement**
- The contribution is the derivation, not the architecture
- We explain *why* decoder-free SAEs work
- No architectural novelty, no state-of-the-art claims
- The contribution is understanding, not performance

**1.5 Roadmap**
- Section 2: Implicit EM and the collapse problem
- Section 3: InfoMax as volume control
- Section 4: The complete objective and architecture
- Section 5: Experiments
- Section 6: Discussion

---

### 2. Background: Implicit EM and the Collapse Problem

**2.1 The Log-Sum-Exp Identity**
- Define component energies E_j(x); lower energy = better explanation
- Define marginal objective: L_LSE = -log Σ_j exp(-E_j)
- Exact identity: ∂L/∂E_j = r_j where r_j = softmax(-E)_j
- The gradient with respect to each energy IS its responsibility

**2.2 Implicit EM**
- No auxiliary E-step computation needed
- Responsibilities emerge in the backward pass
- Parameter updates in optimizer step = M-step
- EM happens implicitly in standard backpropagation

**2.3 Mechanism vs Objective**
- EM defines *how* parameters update (responsibility-weighted)
- EM does not define *what* is learned
- For supervised learning: cross-entropy provides the objective
- For unsupervised learning: what fills this role?

**2.4 The Collapse Problem**
- LSE marginal says "at least one component should explain each input"
- But one component can claim all inputs
- Its responsibility → 1; others receive vanishing gradient
- Other components die; representation degenerates

**2.5 The Mixture Model Analogy**
- Same problem exists in Gaussian mixture models
- GMMs prevent collapse via log-determinant: log det(Σ)
- Penalizes small covariance; component cannot shrink to a point
- Neural LSE objectives have no equivalent term
- Volume control is missing—this is what we must supply

---

### 3. InfoMax as Volume Control

**3.1 Design Requirements**
- Prevent collapse: every component must be active
- Prevent redundancy: components must encode distinct structure
- Inspired by InfoMax principles (Linsker 1988, Bell & Sejnowski 1995)
- No claim of computing mutual information exactly

**3.2 Anti-Collapse: Variance Penalty**
- Dead component has zero variance across dataset
- Penalty: L_var = -Σ_j log Var(A_j)
- Diverges if any component collapses
- Forces every component to respond to some inputs

**3.3 Anti-Redundancy: Decorrelation Penalty**
- Redundant components are correlated
- Penalty: L_tc = ||Corr(A) - I||²
- Forces components to encode different structure
- Approximates independence at second-order statistics

**3.4 Role Equivalence**
- InfoMax plays the role of the log-determinant
- Variance term → prevents dead components (existence)
- Decorrelation term → prevents identical components (diversity)
- Together: volume control for neural implicit EM

---

### 4. The Derived Objective and Architecture

**4.1 Complete Loss Function**
$$L = -\log \sum_j \exp(-E_j) - \lambda_{\text{var}} \sum_j \log \text{Var}(A_j) + \lambda_{\text{tc}} \|\text{Corr}(A) - I\|^2$$

- Term 1 (LSE): EM structure, pulls prototypes toward data
- Term 2 (Variance): Prevents dead components, forces selectivity
- Term 3 (Decorrelation): Prevents redundancy, forces diversity
- Optional: λ_wr ||W^T W - I||² for weight orthogonality

**4.2 Dynamics: Attraction vs Structure**
- LSE is attractive: maximizes likelihood by minimizing energy for matches
- InfoMax is structural: forces selectivity (variance) and diversity (decorrelation)
- Equilibrium: competitive coverage of the data manifold
- Emergent sparsity: no explicit L1 penalty; sparsity arises from competition

**4.3 Architecture**
- z = Wx + b (linear layer)
- E = φ(z) (activation; ReLU, softplus, etc.)
- Responsibilities implicit: r = softmax(-E)
- No decoder, no reconstruction term

**4.4 Energy vs Activation Interpretation**
- Encoder outputs energies (distances): low = close to prototype
- Standard SAEs output activations (similarities): high = close to prototype
- For benchmarking against activation-based methods: s = max(E) - E
- This conversion is for comparison only; not needed for training

**4.5 Implicit Decoder**
- For reconstruction metrics, use W^T as implicit decoder: x̂ = W^T a
- If encoder learns reversible basis, reconstruction emerges without training
- This validates information preservation without explicit reconstruction loss

---

### 5. Experiments

**5.1 Experiment 1: Verifying the Identity**
- Goal: Confirm ∂L_LSE/∂E_j = r_j exactly
- Method: Single forward/backward pass; record gradient and responsibility
- Prediction: Perfect identity line (y = x)
- Output: Figure 1 (scatter plot)
- Time: 30 minutes

**5.2 Experiment 2: Ablation**
- Goal: Show each term is necessary
- Configurations:
  - A: LSE only
  - B: LSE + variance
  - C: LSE + variance + decorrelation (full)
  - D: Variance + decorrelation only (no LSE)
- Metrics: Dead units, redundancy score, feature usage distribution
- Predictions:
  - A: Collapse—one component dominates, others die
  - B: No dead units, but redundant features
  - C: Stable, diverse, non-redundant features
  - D: No EM structure—whitening without competitive dynamics
- Output: Table + Figure 2 (correlation matrices or bar chart)
- Time: 2 hours

**5.3 Experiment 3: Benchmark**
- Goal: Show competitive with standard SAE
- Dataset: MNIST or GPT-2 activations
- Baseline: Standard SAE (encoder + decoder + L1)
- Metrics:
  - Reconstruction MSE (ours uses W^T)
  - Sparsity (L0 norm on activations)
  - Parameter count
- Predictions:
  - Comparable reconstruction quality
  - Similar sparsity levels
  - ~50% fewer parameters (no decoder)
- Output: Table (Figure 3)
- Time: 4 hours

---

### 6. Discussion

**6.1 What We Showed**
- Decoder-free SAEs arise from implicit EM + volume control
- LSE provides competitive dynamics; InfoMax prevents collapse
- The derivation is principled, not heuristic
- Each term has a clear, necessary role

**6.2 What We Did Not Show**
- That this is *better* than existing SAEs (we showed competitive, not superior)
- Large-scale validation on LLMs
- Architectural novelty (the architecture exists; we explain it)

**6.3 Why Decoders Appeared Necessary**
- Reconstruction implicitly provides volume control
- Forces encoder to preserve information (anti-collapse)
- Forces diverse features (anti-redundancy via reconstruction pressure)
- With explicit InfoMax regularization, this role is filled directly
- The decoder was compensatory, not fundamental

**6.4 Limitations**
- Single-layer experiments only
- Limited to MNIST / small scale
- Theoretical claims stronger than empirical validation
- Did not explore all activation functions or architectures

**6.5 Future Work**
- Scale to LLM interpretability applications
- Multi-layer extensions
- Tighter theoretical analysis of convergence
- Connection to other implicit inference methods

---

### 7. Conclusion

Sparse autoencoders have relied on decoders to ensure information preservation. We showed this is unnecessary. The decoder was compensating for missing volume control in the encoder objective. By deriving the objective from implicit EM theory, we identified InfoMax regularization as the missing piece—analogous to the log-determinant in Gaussian mixture models. Log-sum-exp provides competitive dynamics; InfoMax prevents collapse. Together, they yield a complete encoder-only objective. The decoder is a redundant artifact of previous formulations.

---

## Figures

**Figure 1: Theorem Verification**
- Scatter plot: x-axis = r_j (responsibility), y-axis = ∂L/∂E_j (gradient)
- Expected: Perfect y = x diagonal line
- Caption: "The implicit EM identity holds exactly: the gradient with respect to each energy equals its responsibility."

**Figure 2: Ablation Results**
- Option A: 2×2 grid of feature correlation matrices by configuration
- Option B: Grouped bar chart showing dead units and redundancy by configuration
- Caption: "Each regularization term serves a distinct purpose. LSE alone collapses (A). Adding variance prevents death but not redundancy (B). Full objective yields stable, diverse features (C)."

**Figure 3: Benchmark Comparison**
- Table: Model | Recon. MSE | Sparsity (L0) | Parameters
- Or grouped bar chart
- Caption: "Decoder-free objective matches standard SAE reconstruction and sparsity with approximately half the parameters."

---

## Key Equations

1. **LSE Objective:** 
$$L_{\text{LSE}} = -\log \sum_j \exp(-E_j(x))$$

2. **The Identity:** 
$$\frac{\partial L_{\text{LSE}}}{\partial E_j} = r_j = \frac{\exp(-E_j)}{\sum_k \exp(-E_k)}$$

3. **Variance Penalty:** 
$$L_{\text{var}} = -\sum_j \log \text{Var}(A_j)$$

4. **Decorrelation Penalty:** 
$$L_{\text{tc}} = \|\text{Corr}(A) - I\|^2$$

5. **Complete Objective:** 
$$L = L_{\text{LSE}} + \lambda_{\text{var}} L_{\text{var}} + \lambda_{\text{tc}} L_{\text{tc}}$$

6. **Architecture:** 
$$z = Wx + b, \quad E = \phi(z), \quad r = \text{softmax}(-E)$$

---

## Target Length

| Section | Pages |
|---------|-------|
| Abstract | 0.20 |
| 1. Introduction | 0.80 |
| 2. Background + Collapse | 1.25 |
| 3. InfoMax as Volume Control | 0.75 |
| 4. Objective + Architecture | 1.00 |
| 5. Experiments | 1.25 |
| 6. Discussion | 0.75 |
| 7. Conclusion | 0.25 |
| References | 0.50 |
| **Total** | **6.75** |

---

## Changes from Original

| Element | Original | Revised |
|---------|----------|---------|
| Sections | 8 | 7 (merged Background + Collapse) |
| Distance/similarity | Not addressed | Section 4.4 explains conversion |
| Experiment predictions | Implicit | Explicit for each |
| "What we didn't show" | Paragraph | Dedicated subsection |
| Decoder explanation | Brief | Section 6.3 explains compensatory role |
| Emergent sparsity | Not mentioned | Section 4.2 callout |
| Conclusion | Generic | Punchy ("redundant artifact") |
| Abstract | Generic | Includes key predictions |