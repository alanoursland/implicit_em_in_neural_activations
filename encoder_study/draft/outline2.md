Here’s the outline that matches the vision I described — tight, first-principles, and inevitability-driven rather than “we propose a new method.”

I’ll write this as a paper outline, not a slide deck, and I’ll flag the rhetorical purpose of each section implicitly through structure rather than commentary.

---

**Title**
**Deriving Decoder-Free Sparse Autoencoders from First Principles**

---

### 1. Introduction

1.1 Sparse autoencoders as feature discovery tools

* Why SAEs are widely used despite architectural inefficiency
* Decoder cost (parameters, compute) and conceptual awkwardness

1.2 The central question

* What is the decoder *doing* for the encoder?
* Is reconstruction fundamental, or compensatory?

1.3 Observation

* Encoder-only SAEs exist and work, but lack a principled derivation

1.4 Contribution

* A first-principles derivation of a complete encoder-only objective
* No architectural novelty, no state-of-the-art claims
* The contribution is understanding, not performance

---

### 2. Background: Implicit EM and Distance-Based Representations

2.1 Linear layers as distance computations

* Weight vectors as prototype directions
* Bias as prototype location
* Activations as energies or distances

2.2 Log-sum-exp objectives

* Definition of the LSE marginal
* Exact gradient identity: gradient equals responsibility
* Implicit EM via backpropagation

2.3 Mechanism vs objective

* EM defines *how* parameters update
* EM does not define *what* is learned
* Supervised vs unsupervised gap

---

### 3. The Collapse Problem

3.1 Pure log-sum-exp in unsupervised learning

* “At least one component should explain each input”

3.2 Degenerate solutions

* One component claims all inputs
* Others receive vanishing gradients and die

3.3 Classical analogy: Gaussian mixture models

* Role of the log-determinant
* Volume control prevents collapse

3.4 Missing ingredient

* Neural LSE objectives lack volume control
* Collapse is inevitable without additional constraints

---

### 4. InfoMax as Volume Control

4.1 Design requirements

* All components must be active
* Components must encode distinct structure

4.2 InfoMax principle (operational)

* Maximize marginal entropy
* Minimize total correlation

4.3 Practical instantiation

* Variance term: anti-collapse
* Correlation penalty: anti-redundancy

4.4 Role equivalence

* InfoMax plays the role of the log-determinant
* Prevents both dead and redundant components

---

### 5. The Derived Objective

5.1 Complete loss function

[
L = -\log \sum_j \exp(-a_j)
;-;
\lambda_{\text{var}} \sum_j \log \mathrm{Var}(a_j)
;+;
\lambda_{\text{tc}} |\mathrm{Corr}(A) - I|^2
]

5.2 Interpretation of each term

* LSE: EM structure and competition
* Variance: enforce component usage
* Decorrelation: enforce diversity

5.3 Architecture

* Single linear encoder
* Nonnegative activation
* No decoder, no reconstruction loss

---

### 6. Why This Works

6.1 Competing forces

* LSE as repulsive force
* InfoMax as attractive force

6.2 Equilibrium dynamics

* Components spread to cover data
* Each specializes in a distinct region

6.3 Relation to competitive learning

* Differentiable vector quantization
* Soft responsibilities instead of hard assignments

6.4 Emergent sparsity

* No explicit L1 penalty
* Sparsity arises from competition

---

### 7. Experiments

7.1 Experiment 1: Verifying implicit EM

* Gradient vs responsibility identity
* Single forward/backward pass

7.2 Experiment 2: Ablation study

* LSE only
* LSE + variance
* LSE + variance + decorrelation
* InfoMax only

7.3 Experiment 3: Benchmark comparison

* Comparison to standard sparse autoencoder
* Reconstruction via transposed weights
* Parameter efficiency

---

### 8. Connections to Prior Work

8.1 Energy-based models

* LSE as negative log marginal likelihood

8.2 Sparse coding

* Reconstruction vs competition-based sparsity

8.3 InfoMax and ICA

* Independence as a representation objective

8.4 Competitive learning and SOMs

* Differentiable, probabilistic formulation

---

### 9. What This Does and Does Not Claim

9.1 What we claim

* Decoder-free SAEs have a principled derivation
* InfoMax supplies missing volume control
* Encoder-only training is sufficient

9.2 What we do not claim

* Architectural novelty
* State-of-the-art performance
* Complete interpretability solution

---

### 10. Discussion

10.1 Why decoders appeared necessary

* Reconstruction as implicit volume control

10.2 Implications for representation learning

* Objective matters more than architecture

10.3 Open questions

* Scaling
* Depth and composition
* Alternative volume regularizers

---

### 11. Conclusion

* Decoder-free sparse autoencoders emerge naturally from implicit EM
* Log-sum-exp provides competition
* InfoMax provides volume control
* With both, the encoder objective is complete

---

If you want, the next natural step would be:

* rewriting this outline in *ICLR-style prose*, or
* identifying which sections must be mathematically formal vs narrative, or
* stress-testing where reviewers are most likely to push back.
