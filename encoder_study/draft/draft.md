# Deriving Decoder-Free Sparse Autoencoders from First Principles

## Abstract

---

## 1. Introduction

### **1.1 The Decoder Tax**
- Sparse autoencoders extract interpretable features from neural networks
- Standard architecture: encoder + decoder + L1 sparsity penalty
- The decoder doubles parameters and compute
- Reconstruction ensures information preservation—but is it fundamental or compensatory?

### **1.2 The Central Question**
- Encoder-only SAEs exist and work empirically
- But they lack principled derivation
- What is the decoder actually *doing* for the encoder?

### **1.3 This Paper**
- We derive decoder-free SAEs from first principles via implicit EM
- Log-sum-exp objectives perform EM implicitly (gradient = responsibility)
- But LSE alone collapses—missing volume control
- InfoMax regularization provides the missing piece
- Result: principled encoder-only objective, no reconstruction needed

### **1.4 Contribution Statement**
- The contribution is the derivation, not the architecture
- We explain *why* decoder-free SAEs work
- No architectural novelty, no state-of-the-art claims
- The contribution is understanding, not performance

### **1.5 Roadmap**
- Section 2: Implicit EM and the collapse problem
- Section 3: InfoMax as volume control
- Section 4: The complete objective and architecture
- Section 5: Experiments
- Section 6: Discussion

---

## 2. Background: Implicit EM and the Collapse Problem

This section develops the theoretical foundation for our derivation. We begin with a key identity: for log-sum-exp objectives, the gradient with respect to each component equals its responsibility (Section 2.1). This identity implies that gradient descent performs expectation-maximization implicitly (Section 2.2). However, this mechanism alone does not determine what is learned (Section 2.3). Without additional constraints, the LSE objective admits degenerate solutions where components collapse (Section 2.4). We draw an analogy to Gaussian mixture models, where the log-determinant prevents such collapse, and observe that neural objectives lack an equivalent term (Section 2.5). This missing piece—volume control—is what we must supply.

### 2.1 The Log-Sum-Exp Identity

Consider an encoder that maps an input $x$ to $K$ component energies $E_1(x), \ldots, E_K(x)$. Following standard conventions in energy-based models (LeCun et al., 2006), we adopt the principle that **lower energy means better explanation**: a component with low energy for a given input is one that "matches" or "claims" that input.

Given these energies, define the log-sum-exp (LSE) marginal objective:

$$L_{\text{LSE}}(x) = -\log \sum_{j=1}^{K} \exp(-E_j(x)) \tag{1}$$

This objective has a natural interpretation: it is minimized when at least one component has low energy for the input. It encodes the requirement that *someone* must explain each data point—the same intuition underlying mixture models (Bishop, 2006).

The key property of this objective is an exact algebraic identity (Oursland, 2025). Taking the gradient with respect to any component energy:

$$\frac{\partial L_{\text{LSE}}}{\partial E_j} = \frac{\exp(-E_j)}{\sum_{k=1}^{K} \exp(-E_k)} = r_j \tag{2}$$

where $r_j$ is precisely the softmax responsibility—the posterior probability that component $j$ explains the input, given the current energies.

This identity is not an approximation. The gradient with respect to each component energy *is* its responsibility. No auxiliary computation is required; the responsibilities emerge directly from the structure of the loss function.

### 2.2 Implicit EM

The identity in Equation 2 has a striking consequence: gradient descent on log-sum-exp objectives performs expectation-maximization implicitly.

In classical EM (Dempster et al., 1977), the E-step computes responsibilities—the posterior probability that each component generated each data point—and the M-step updates parameters using responsibility-weighted sufficient statistics. These two steps alternate explicitly.

With LSE objectives, this separation dissolves. The backward pass computes gradients, but by Equation 2, these gradients *are* the responsibilities. No separate E-step is needed; the responsibilities emerge automatically from backpropagation. The subsequent optimizer update—adjusting parameters in the direction of the gradient—constitutes the M-step, moving each component according to its responsibility for each input.

This implicit EM structure was identified by Oursland (2025), who showed that the gradient identity holds for any parameterization of the energies $E_j(x)$. Whether the encoder is a linear layer, a deep network, or any differentiable function, the same property holds: gradients flowing back through the LSE loss carry responsibility information.

The practical implication is significant. Standard neural network training—forward pass, backward pass, optimizer step—already implements EM dynamics when the loss has LSE structure. Cross-entropy classification is one example: the softmax probabilities that appear in the gradient are precisely the responsibilities in a categorical mixture. The question for unsupervised learning is: what objective provides the analogous structure?

### 2.3 Mechanism vs Objective

It is important to distinguish between what implicit EM *provides* and what it *requires*.

The LSE identity (Equation 2) defines a **mechanism**: parameters update according to responsibility-weighted gradients. Components that claim more responsibility for an input receive larger gradient contributions from that input. This is *how* learning proceeds—but it does not specify *what* is learned.

For supervised learning, the objective is clear. Cross-entropy loss has LSE structure, and the target labels define what the model should learn. The responsibilities (softmax outputs) are trained to match the labels. The mechanism (implicit EM) and the objective (match the targets) work together.

For unsupervised learning, the situation is different. There are no labels. The LSE marginal in Equation 1 provides a candidate objective: minimize the energy of the best-matching component for each input. Intuitively, this says "at least one component should explain each data point." But this objective alone, without additional constraints, admits degenerate solutions.

The mechanism is in place. What remains is to identify the constraints that make it useful—the volume control that prevents collapse while allowing the implicit EM dynamics to discover meaningful structure.

### 2.4 The Collapse Problem

The LSE objective in Equation 1 encodes a reasonable requirement: for each input, at least one component should have low energy. But this requirement is too weak. It admits a degenerate solution that renders the representation useless.

Consider what happens if a single component $j^*$ learns to produce low energy for all inputs. Its responsibility $r_{j^*} \to 1$ across the dataset, while all other responsibilities $r_{j \neq j^*} \to 0$. By Equation 2, the gradient with respect to other components vanishes. They receive no learning signal. They die.

The result is a collapsed representation: one component that responds to everything, and $K-1$ components that respond to nothing. The encoder has satisfied the LSE objective—every input is explained—but the representation carries no more information than a constant.

This failure mode is not unique to neural networks. The same problem arises in Gaussian mixture models, where a single component can expand to cover the entire dataset while others shrink to zero variance (Bishop, 2006). In that setting, the log-determinant of the covariance matrix acts as a regularizer, penalizing components that become too diffuse or too concentrated. A component cannot collapse to a point or expand without bound.

Neural LSE objectives have no such term. The implicit EM machinery is present—Equation 2 gives us responsibility-weighted updates—but the volume control is missing. Without it, collapse is not merely possible; given sufficient capacity and training time, it is likely. Something must fill the role that the log-determinant plays in mixture models.

### 2.5 The Mixture Model Analogy

The collapse problem is not new. Gaussian mixture models face the same challenge, and their solution illuminates what neural objectives lack.

In a GMM, the log-likelihood of a data point under component $k$ with mean $\mu_k$ and covariance $\Sigma_k$ is:

$$\log P(x \mid k) = -\frac{1}{2}(x - \mu_k)^\top \Sigma_k^{-1}(x - \mu_k) - \frac{1}{2}\log\det(\Sigma_k) + \text{const} \tag{3}$$

The first term is the Mahalanobis distance—how far the point lies from the component center, scaled by the covariance. The second term, the log-determinant, is crucial: it penalizes components with small covariance (Bishop, 2006).

Without the log-determinant, a component could collapse to a single point, achieving arbitrarily high likelihood for any data point it sits on. With it, shrinking the covariance incurs an unbounded penalty. The log-determinant enforces a kind of "volume"—each component must maintain a region of support, not a singularity.

Neural energy models have no equivalent term. The encoder computes energies $E_j(x)$, and the LSE loss aggregates them, but nothing constrains the *distribution* of those energies across the dataset. A component can produce uniformly low energy (claiming everything) or uniformly high energy (claiming nothing) without penalty.

The log-determinant serves as **volume control**: it ensures that each component maintains meaningful extent in the data space. For neural implicit EM to work, we must supply an analogous constraint—something that prevents components from collapsing to triviality while allowing the responsibility-weighted learning dynamics to discover useful structure.

---

## 3. InfoMax as Volume Control

Section 2 established that implicit EM provides a learning mechanism—responsibility-weighted updates—but lacks the constraints needed to prevent collapse. Gaussian mixture models solve this problem with the log-determinant; neural networks require an equivalent.

This section introduces InfoMax regularization as that equivalent. We first specify the design requirements: components must be active and distinct (Section 3.1). We then present two penalties that enforce these requirements: a variance penalty to prevent collapse (Section 3.2) and a decorrelation penalty to prevent redundancy (Section 3.3). Finally, we show that these penalties together play the same role as the log-determinant in mixture models (Section 3.4). The result is a complete volume control mechanism for neural implicit EM.

### 3.1 Design Requirements

What properties must a volume control regularizer satisfy? We identify two requirements, both necessary to prevent the failure modes described in Section 2.4.

**Requirement 1: Prevent collapse.** Every component must be active across the dataset. A component that never fires—or fires uniformly for all inputs—carries no information. The regularizer must ensure that each component responds selectively: high responsibility for some inputs, low for others.

**Requirement 2: Prevent redundancy.** Components must encode distinct structure. If two components produce identical energies across all inputs, one is superfluous. The regularizer must encourage diversity: different components should capture different aspects of the data.

These requirements echo the principles of information maximization. Linsker (1988) showed that maximizing information transmission in neural networks leads to representations with high entropy and low redundancy. Bell and Sejnowski (1995) operationalized this as maximizing the joint entropy of outputs while minimizing their mutual information—equivalently, maximizing marginal entropies while encouraging independence.

We draw inspiration from this principle, but make no claim of computing mutual information exactly. The InfoMax literature provides the conceptual vocabulary—entropy, independence, redundancy—and motivates the form of our regularizer. What we require is not a precise information-theoretic quantity, but an operational proxy that enforces the two requirements above: keep components active, keep them distinct.

### 3.2 Anti-Collapse: Variance Penalty

The first requirement—preventing collapse—can be enforced by penalizing components with low variance across the dataset.

Consider the activations $A_j = \{a_j(x_1), \ldots, a_j(x_N)\}$ of component $j$ over $N$ data points. If component $j$ has collapsed—producing the same output for all inputs—then $\text{Var}(A_j) = 0$. Conversely, a component that responds selectively, with high activation for some inputs and low for others, will have high variance.

We penalize low variance with a logarithmic barrier:

$$L_{\text{var}} = -\sum_{j=1}^{K} \log \text{Var}(A_j) \tag{4}$$

This penalty has two important properties. First, it diverges as any component's variance approaches zero: $\text{Var}(A_j) \to 0$ implies $L_{\text{var}} \to \infty$. Collapse is not merely discouraged; it is forbidden. Second, the logarithm provides diminishing returns for increasing variance, preventing any single component from dominating the objective by inflating its variance without bound.

The form of Equation 4 is not arbitrary. Under a Gaussian assumption, the entropy of a random variable is $\frac{1}{2}\log(2\pi e \cdot \text{Var})$. Maximizing entropy is equivalent to maximizing log-variance. Our penalty is thus an entropy proxy: it encourages each component to have high marginal entropy, which in the InfoMax framework (Linsker, 1988) corresponds to transmitting maximal information.

### 3.3 Anti-Redundancy: Decorrelation Penalty

The second requirement—preventing redundancy—can be enforced by penalizing correlations between components.

If two components produce identical activations across the dataset, they are perfectly correlated. More generally, if component $i$ can be predicted from component $j$ via a linear relationship, the representation contains redundancy: the two components encode overlapping information. True independence implies zero correlation, though the converse does not hold.

We penalize deviation from uncorrelated outputs:

$$L_{\text{tc}} = \|\text{Corr}(A) - I\|_F^2 \tag{5}$$

where $\text{Corr}(A)$ is the $K \times K$ correlation matrix computed over the dataset, $I$ is the identity matrix, and $\|\cdot\|_F$ denotes the Frobenius norm. This penalty is zero when all components are uncorrelated (diagonal correlation matrix) and grows as off-diagonal correlations increase.

The decorrelation penalty enforces independence at the level of second-order statistics. Full statistical independence requires matching all moments, but for many distributions—and particularly under Gaussian assumptions—decorrelation is a reasonable proxy. Bell and Sejnowski (1995) showed that decorrelation combined with entropy maximization yields independent components for a broad class of distributions.

This same regularizer appears in recent self-supervised learning methods. Barlow Twins (Zbontar et al., 2021) uses a cross-correlation loss between augmented views, and VICReg (Bardes et al., 2022) explicitly decomposes its objective into variance, invariance, and covariance terms. Our decorrelation penalty is equivalent to their covariance term. The convergent appearance of this regularizer across different contexts suggests it captures something fundamental about preventing representational collapse.

### 3.4 Role Equivalence

We can now make the analogy to mixture models precise. The variance and decorrelation penalties together play the role that the log-determinant plays in Gaussian mixture models.

Recall from Section 2.5 that the log-determinant $\log\det(\Sigma)$ prevents GMM collapse by penalizing components with small covariance. For a multivariate Gaussian, the log-determinant decomposes as the sum of log-eigenvalues of the covariance matrix. When outputs are uncorrelated—so the covariance matrix is diagonal—this reduces to:

$$\log\det(\Sigma) = \sum_{j=1}^{K} \log \text{Var}(A_j) \tag{6}$$

This is precisely the negative of our variance penalty (Equation 4). Maximizing the log-determinant is equivalent to minimizing $L_{\text{var}}$ when off-diagonal covariances are zero.

The decorrelation penalty (Equation 5) enforces the condition that makes this equivalence hold. By driving the correlation matrix toward the identity, $L_{\text{tc}}$ ensures that the covariance structure is (approximately) diagonal. The variance term then controls the diagonal entries.

Together, the two penalties constrain the full covariance structure of the activations:

| GMM Term | Neural Equivalent | Function |
|----------|-------------------|----------|
| log det(Σ) diagonal | $L_{\text{var}}$ | Prevent dead components |
| log det(Σ) off-diagonal | $L_{\text{tc}}$ | Prevent redundant components |

This is volume control for neural implicit EM. The variance term ensures each component maintains a region of selective response (existence). The decorrelation term ensures these regions do not overlap (diversity). With both in place, the implicit EM dynamics of Section 2.2 can operate without collapsing to degeneracy.

---

## 4. The Derived Objective and Architecture

We now have the pieces needed to assemble a complete encoder-only objective. Section 2 provided the mechanism: log-sum-exp objectives yield responsibility-weighted gradients, implementing implicit EM. Section 3 provided the constraints: variance and decorrelation penalties supply volume control, preventing collapse and redundancy.

This section combines these elements into a unified framework. We present the complete loss function (Section 4.1) and analyze the dynamics that emerge from the interaction between attraction and structure (Section 4.2). We then specify the minimal architecture required (Section 4.3), clarify the relationship between our energy-based formulation and standard activation-based SAEs (Section 4.4), and show how reconstruction quality can be evaluated without an explicit decoder (Section 4.5). The result is a principled, decoder-free sparse autoencoder derived entirely from the considerations above.

### 4.1 Complete Loss Function

Combining the implicit EM structure from Section 2 with the volume control from Section 3, we arrive at the complete objective:

$$L = -\log \sum_{j=1}^{K} \exp(-E_j) - \lambda_{\text{var}} \sum_{j=1}^{K} \log \text{Var}(A_j) + \lambda_{\text{tc}} \|\text{Corr}(A) - I\|_F^2 \tag{7}$$

Each term serves a distinct purpose:

**Term 1: Log-sum-exp.** The LSE term provides implicit EM structure (Section 2.2). Its gradient with respect to each energy equals the responsibility (Equation 2), yielding responsibility-weighted parameter updates. This term pulls components toward the data—minimizing the energy of the best-matching component for each input.

**Term 2: Variance penalty.** This term prevents collapse (Section 3.2). By penalizing components with low variance, it ensures every component responds selectively to some subset of inputs. The hyperparameter $\lambda_{\text{var}}$ controls the strength of this constraint.

**Term 3: Decorrelation penalty.** This term prevents redundancy (Section 3.3). By penalizing correlated outputs, it forces components to encode distinct structure. The hyperparameter $\lambda_{\text{tc}}$ controls the strength of this constraint.

Notably, no reconstruction term appears. Information preservation is not enforced by matching a decoder's output to the input, but by the combination of competitive dynamics (LSE) and volume control (InfoMax).

For some architectures, an additional weight regularization term may be beneficial:

$$L_{\text{wr}} = \lambda_{\text{wr}} \|W^\top W - I\|_F^2 \tag{8}$$

This penalty encourages orthogonality among the encoder's weight vectors, preventing multiple components from learning identical directions in input space (Oursland, 2024). We treat this as optional; our primary results use only Equation 7.

### 4.2 Dynamics: Attraction vs Structure

The objective in Equation 7 creates a tension between two forces that together produce useful representations.

**The LSE term is attractive.** Minimizing $-\log \sum_j \exp(-E_j)$ is equivalent to maximizing the likelihood that at least one component explains each input. This pulls components toward the data: a component reduces its contribution to the loss by lowering its energy for inputs it can explain well. Left unchecked, this attraction leads to collapse—one component claims everything.

**The InfoMax terms are structural.** The variance penalty (Equation 4) forces each component to be selective—responding strongly to some inputs and weakly to others. The decorrelation penalty (Equation 5) forces components to be diverse—capturing different aspects of the data. Together, they constrain *how* components can reduce their energy, without dictating *where* they should go.

**The equilibrium is competitive coverage.** Under these opposing forces, the system settles into a state where components tile the data manifold. Each component specializes in a region of input space, achieving low energy (high responsibility) for inputs in its region and high energy (low responsibility) elsewhere. This resembles the competitive dynamics in self-organizing maps (Kohonen, 1982), but with soft responsibilities rather than hard winner-take-all assignments.

**Sparsity is emergent.** Equation 7 contains no explicit sparsity penalty—no L1 norm on activations. Yet sparse representations arise naturally. When components specialize, most inputs activate only a few components strongly; the rest have high energy and near-zero responsibility. Sparsity is not imposed but emerges from competition. This contrasts with standard sparse autoencoders, where L1 regularization forces sparsity regardless of whether the data supports it.

### 4.3 Architecture

The objective in Equation 7 can be minimized with a minimal architecture: a single linear layer followed by a nonlinearity.

$$z = Wx + b, \qquad E = \phi(z) \tag{9}$$

Here $W \in \mathbb{R}^{K \times D}$ maps inputs $x \in \mathbb{R}^D$ to $K$ pre-activation values $z$, the bias $b \in \mathbb{R}^K$ shifts each component, and the activation function $\phi$ produces the final energies $E$.

The choice of activation affects the geometry of the energy landscape but not the implicit EM property (Equation 2), which holds for any differentiable parameterization. We consider several options:

- **ReLU:** $\phi(z) = \max(0, z)$. Produces non-negative energies. Inputs with $z_j < 0$ have zero energy for component $j$—strong matches. Introduces sparsity in the energy representation.
- **Softplus:** $\phi(z) = \log(1 + \exp(z))$. A smooth approximation to ReLU. Non-negative energies without the discontinuous gradient at zero.
- **Identity:** $\phi(z) = z$. Energies can be positive or negative. Simplest case; useful for analysis.

The responsibilities need not be computed explicitly during training. By Equation 2, they appear implicitly in the gradients:

$$r = \text{softmax}(-E) \tag{10}$$

This expression is useful for analysis and visualization, but the training procedure requires only standard backpropagation through Equation 7.

Critically, the architecture contains no decoder. There is no matrix $W'$ mapping activations back to input space, and no reconstruction loss $\|x - W'a\|^2$. The encoder alone constitutes the model. This is the decoder-free sparse autoencoder promised in Section 1—derived not as an architectural choice, but as a consequence of the objective.

### 4.4 Energy vs Activation Interpretation

Our formulation outputs energies; standard sparse autoencoders output activations. These are inverse conventions that must be reconciled for comparison.

**Energy convention (this work).** The encoder produces energies $E_j(x)$, where lower values indicate better matches. A component with low energy for an input "claims" that input with high responsibility. This follows the convention in energy-based models (LeCun et al., 2006) and mixture models, where probability increases as energy decreases.

**Activation convention (standard SAEs).** Traditional sparse autoencoders produce activations $a_j(x)$, where higher values indicate stronger feature presence. A component with high activation for an input is considered "active" for that input. This follows the convention in neural networks, where ReLU outputs are interpreted as feature detectors.

The two conventions are related by negation. High activation corresponds to low energy; a strongly active feature is one with a strong match.

For benchmarking against activation-based methods, we convert energies to activations:

$$s = E_{\max} - E \tag{11}$$

where $E_{\max} = \max_j E_j$ within each input. This transformation preserves ordering—the component with lowest energy has highest activation—while shifting the range so that the "best match" has activation near zero energy corresponds to high activation.

This conversion is for comparison only. During training, we work entirely with energies and the objective in Equation 7. The conversion is applied post-hoc when computing metrics like sparsity (L0 norm) that are conventionally defined on activations, or when comparing learned features to those from standard SAEs.

### 4.5 Implicit Decoder

Although our objective contains no reconstruction term, we can still evaluate reconstruction quality—and doing so provides a test of whether the encoder preserves information about the input.

Given activations $a$ (converted from energies via Equation 11 or taken as $a = \phi(z)$ directly), we reconstruct the input using the transposed encoder weights:

$$\hat{x} = W^\top a \tag{12}$$

This is not a learned decoder. The matrix $W^\top$ is simply the transpose of the encoder weights; no additional parameters are introduced, and no reconstruction loss influences training.

Why might this work? If the encoder learns weight vectors that form a (approximately) orthonormal basis—encouraged by the decorrelation penalty and optional weight regularization (Equation 8)—then the encoding is approximately invertible. The rows of $W$ act as analysis filters; the columns of $W^\top$ act as synthesis filters. This is the structure of classical linear transforms like PCA and ICA, where the same matrix (or its inverse) serves for both encoding and decoding.

We use Equation 12 to compute reconstruction error for benchmarking against standard SAEs (Section 5.3). If the reconstruction quality is competitive despite never training on reconstruction, this validates the central claim: the decoder in standard SAEs is compensatory. It enforces information preservation indirectly through reconstruction pressure. Our objective enforces the same property directly through volume control—and the implicit decoder $W^\top$ reveals that the information was preserved all along.

---

## 5. Experiments

### **5.1 Experiment 1: Verifying the Identity**
- Goal: Confirm ∂L_LSE/∂E_j = r_j exactly
- Method: Single forward/backward pass; record gradient and responsibility
- Prediction: Perfect identity line (y = x)
- Output: Figure 1 (scatter plot)
- Time: 30 minutes

### **5.2 Experiment 2: Ablation**
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

### **5.3 Experiment 3: Benchmark**
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

## 6. Discussion

### **6.1 What We Showed**
- Decoder-free SAEs arise from implicit EM + volume control
- LSE provides competitive dynamics; InfoMax prevents collapse
- The derivation is principled, not heuristic
- Each term has a clear, necessary role

### **6.2 What We Did Not Show**
- That this is *better* than existing SAEs (we showed competitive, not superior)
- Large-scale validation on LLMs
- Architectural novelty (the architecture exists; we explain it)

### **6.3 Why Decoders Appeared Necessary**
- Reconstruction implicitly provides volume control
- Forces encoder to preserve information (anti-collapse)
- Forces diverse features (anti-redundancy via reconstruction pressure)
- With explicit InfoMax regularization, this role is filled directly
- The decoder was compensatory, not fundamental

### **6.4 Limitations**
- Single-layer experiments only
- Limited to MNIST / small scale
- Theoretical claims stronger than empirical validation
- Did not explore all activation functions or architectures

### **6.5 Future Work**
- Scale to LLM interpretability applications
- Multi-layer extensions
- Tighter theoretical analysis of convergence
- Connection to other implicit inference methods

---

## 7. Conclusion

Sparse autoencoders have relied on decoders to ensure information preservation. We showed this is unnecessary. The decoder was compensating for missing volume control in the encoder objective. By deriving the objective from implicit EM theory, we identified InfoMax regularization as the missing piece—analogous to the log-determinant in Gaussian mixture models. Log-sum-exp provides competitive dynamics; InfoMax prevents collapse. Together, they yield a complete encoder-only objective. The decoder is a redundant artifact of previous formulations.

---

# Figures

### **Figure 1: Theorem Verification**
- Scatter plot: x-axis = r_j (responsibility), y-axis = ∂L/∂E_j (gradient)
- Expected: Perfect y = x diagonal line
- Caption: "The implicit EM identity holds exactly: the gradient with respect to each energy equals its responsibility."

### **Figure 2: Ablation Results**
- Option A: 2×2 grid of feature correlation matrices by configuration
- Option B: Grouped bar chart showing dead units and redundancy by configuration
- Caption: "Each regularization term serves a distinct purpose. LSE alone collapses (A). Adding variance prevents death but not redundancy (B). Full objective yields stable, diverse features (C)."

### **Figure 3: Benchmark Comparison**
- Table: Model | Recon. MSE | Sparsity (L0) | Parameters
- Or grouped bar chart
- Caption: "Decoder-free objective matches standard SAE reconstruction and sparsity with approximately half the parameters."

---

# Key Equations

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

# Target Length

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

# Changes from Original

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