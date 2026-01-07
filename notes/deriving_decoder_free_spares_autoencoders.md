# Deriving Decoder-Free Sparse Autoencoders from Implicit EM

## Introduction

This document derives a neural architecture from first principles using the theory of implicit expectation-maximization (EM) in gradient descent. The result is a model equivalent to decoder-free sparse autoencoders and closely related to energy-based models, but arrived at through a different path: the algebraic structure of log-sum-exp objectives.

The derivation follows from two prior results:

1. **Mahalanobis interpretation (Oursland 2024):** Linear layers compute distances to prototypes. Neurons learn principal components of Gaussian clusters. Orthogonality regularization encourages statistically independent features.

2. **Implicit EM (Oursland 2025):** For log-sum-exp objectives, the gradient with respect to each component equals its responsibility. Gradient descent performs EM implicitly.

Combining these insights leads to a principled unsupervised objective: log-sum-exp for EM structure, plus regularization for volume control.

## Sign Convention

Throughout this document, we adopt the convention that **lower energy means better explanation**. A component with low energy for an input claims high responsibility for that input. This convention, stated once here, resolves all sign ambiguities that follow.

## Implicit EM Requires Log-Sum-Exp

Consider a set of component energies $E_j(x)$, where lower energy means better explanation of the input. Define the marginal objective:

$$L_{\text{LSE}}(x) = -\log \sum_j \exp(-E_j(x))$$

This objective has an exact algebraic property:

$$\frac{\partial L_{\text{LSE}}}{\partial E_j(x)} = r_j(x), \qquad r_j(x) = \frac{\exp(-E_j(x))}{\sum_k \exp(-E_k(x))}$$

The gradient with respect to each component energy is its responsibility. No auxiliary variables are introduced; the E-step appears implicitly in backpropagation. Gradient descent on this objective therefore performs responsibility-weighted updates exactly analogous to EM.

This identity holds regardless of how the energies $E_j(x)$ are parameterized.

For supervised learning, cross-entropy provides this structure:

$$L = -\log P(y|x) = -z_y + \log \sum_k \exp(z_k)$$

For unsupervised learning, what is the objective? The log-sum-exp marginal is a natural choice: it says that at least one component should explain each input.

## The Collapse Problem (Missing Volume Control)

The log-sum-exp marginal alone is insufficient for unsupervised learning. A single component can lower its energy for all inputs, causing its responsibility to approach one while all other components receive vanishing gradient and die.

This is the neural analogue of a known issue in mixture models: without explicit control over component "volume" or effective support, maximum-likelihood objectives admit degenerate solutions.

In classical Gaussian mixture models, the log-determinant prevents collapse:

$$\log P(x|k) \propto -\frac{1}{2}(x-\mu_k)^\top\Sigma_k^{-1}(x-\mu_k) - \frac{1}{2} \log \det(\Sigma_k)$$

The log-determinant penalizes small covariance. A component cannot shrink to a point.

In neural energy models, no such term exists by default. Therefore, any practical implicit-EM objective must include an explicit volume control regularizer.

## Volume Control via Regularization (InfoMax-Inspired)

We introduce a regularizer whose sole purpose is to prevent collapse and redundancy. We refer to this bundle as **InfoMax regularization**, defined operationally as:

1. **Anti-collapse:** Every component must be active across the dataset.
2. **Redundancy reduction:** Distinct components must encode distinct structure.

No claim is made that this computes mutual information exactly. The name reflects the inspiration from information-maximization principles (Linsker 1988, Bell & Sejnowski 1995), not an assertion of mathematical equivalence.

### Anti-collapse (Activity Volume)

A component that never responds has zero effective support. We penalize this by encouraging nonzero dispersion across the dataset:

$$L_{\text{var}} = -\sum_j \log \text{Var}(A_j)$$

If a component collapses, its variance goes to zero and the penalty diverges.

### Redundancy Reduction

To prevent multiple components from encoding the same structure, we penalize second-order dependence:

$$L_{\text{tc}} = \|\text{Corr}(A) - I\|^2$$

This encourages decorrelated outputs and approximates independence at the level of linear statistics.

Together, these terms enforce effective volume and diversity in representation space, serving the role that the log-determinant plays in Gaussian mixture models.

## The Sign-Flip: A Min-Max Game

There is a crucial sign difference between standard clustering and our model that reveals the dynamics at play.

**Standard clustering EM:**
$$\max \log \sum_j \exp(z_j) \rightarrow \text{pull prototypes toward data}$$

**Our model:**
$$\min \log \sum_j \exp(z_j) \rightarrow \text{push prototypes apart}$$

This inverts the role of the LSE term. Minimizing LSE penalizes prototypes that are too close to inputs relative to others. It acts as a repulsive force that distributes responsibility.

**InfoMax is the attractive force.** The variance term $-\sum_j \log \text{Var}(A_j)$ rewards prototypes that respond strongly to inputs. To have high variance, a prototype must be close to some data. InfoMax pulls prototypes toward the data.

**The model is a min-max game:**

| Term | Force | Effect |
|------|-------|--------|
| InfoMax (variance) | Attractive | Pull prototypes toward data |
| LSE | Repulsive | Push prototypes apart (distribute responsibility) |

This implements competitive learning:
- Every prototype tries to get close to data (InfoMax)
- LSE taxes prototypes proportional to their responsibility
- The equilibrium: prototypes spread out to cover the data

This is differentiable vector quantization. Similar dynamics appear in Self-Organizing Maps (Kohonen) and competitive learning networks. The difference: our formulation is fully differentiable and probabilistic, with soft responsibilities emerging via the implicit softmax in the LSE gradient.

## Complete Objective

The full loss is:

$$L = -\log \sum_j \exp(-E_j(x)) + \lambda_{\text{var}} \left(-\sum_j \log \text{Var}(A_j)\right) + \lambda_{\text{tc}} \|\text{Corr}(A) - I\|^2 + \lambda_{\text{wr}} \|W^\top W - I\|^2$$

**Term 1 — Log-sum-exp:** Provides implicit EM structure via responsibility-weighted gradients. The repulsive force that distributes components.

**Term 2 — Variance:** Prevents dead components. The attractive force that keeps prototypes engaged with data.

**Term 3 — Correlation penalty:** Prevents redundant output representations.

**Term 4 — Weight regularization:** Prevents degenerate or duplicate parameterizations. (This term was predicted in the Mahalanobis paper as "orthogonality constraint or regularization on the weight matrices.")

This objective is decoder-free: no reconstruction term appears.

## Architecture

A minimal instantiation is:

$$z = Wx + b, \qquad E = \phi(z)$$

where $\phi$ is any activation producing component energies.

Lower energy corresponds to better explanation. Responsibilities are computed implicitly via:

$$r = \text{softmax}(-E)$$

The choice of $\phi$ affects geometry but not the implicit-EM property. ReLU, softplus, or identity may be used, provided scale is controlled.

**Concrete example:**

```
z = Wx + b
a = relu(z)
L = -log Σⱼ exp(-aⱼ) + λ_var(-Σⱼ log Var(aⱼ)) + λ_tc ||Corr(A) - I||² + λ_wr ||WᵀW - I||²
```

**Linear layer:** Computes distances to prototypes. Each row of W is a prototype direction.

**ReLU:** Ensures non-negative energies. Introduces sparsity. Low activation = high energy = far from prototype.

**Combined loss:** EM structure plus volume control.

No decoder. No reconstruction. The objective is implicit likelihood with information-theoretic regularization.

## Interpretation

- The log-sum-exp term implements maximum likelihood under a mixture of energy-based components and induces EM-style competition.
- The regularization terms supply the missing volume control required for stable unsupervised learning.
- Sparsity emerges from competitive responsibility allocation rather than explicit $L_1$ penalties.
- No decoder is required: information is preserved by construction through responsibility competition and volume constraints.

## Empirical Observations

Experiments on MNIST with a single-layer model revealed several findings:

**Without InfoMax (pure logsumexp):** Collapse occurs. One prototype dominates all inputs.

**With variance term only:** Collapse prevented. All units remain active. But output redundancy is possible.

**With variance + decorrelation:** Non-redundant outputs achieved. However, weights can still be redundant via a "bias loophole"—with ReLU, identical weight vectors with different biases can produce independent outputs by firing on different input subsets.

**With weight redundancy penalty:** Forces diverse weight directions. Closes the bias loophole.

**Softmax in forward pass:** When softmax appears as an explicit activation (not just implicit in the loss gradient), competition provides implicit regularization. Weight vectors naturally differentiate. The need for explicit $\lambda_{\text{tc}}$ and $\lambda_{\text{wr}}$ is reduced.

**Optimizer behavior:** With explicit softmax activation, SGD and Adam perform similarly. Without it, Adam significantly outperforms SGD. This suggests EM structure normalizes the optimization landscape.

| Configuration | Needs $\lambda_{\text{tc}}$ | Needs $\lambda_{\text{wr}}$ |
|---------------|-----------|-----------|
| Linear → ReLU → LogSumExp loss | Yes | Yes |
| Linear → ReLU → Softmax → loss | Less | Less |

**Recommendation:** Start with full regularization ($\lambda_{\text{tc}} > 0$, $\lambda_{\text{wr}} > 0$). Ablate to find minimal necessary terms for your architecture.

## Connection to Prior Work

### Energy-Based Models (LeCun et al. 2006)

EBMs define probability via energy:

$$P(x) \propto \exp(-E(x))$$

Our logsumexp term is the negative log marginal likelihood under a mixture of energy functions:

$$L = -\log \sum_j \exp(-E_j(x))$$

Each prototype $j$ defines an energy $E_j(x) = \phi(w_j^\top x + b_j)$.

The InfoMax terms are regularization on the energy landscape—ensuring diverse, well-spread energy wells.

### Sparse Coding (Olshausen & Field 1996)

Sparse coding learns:

$$\min \|x - Da\|^2 + \lambda\|a\|_1$$

This requires a decoder $D$ and explicit sparsity penalty.

Our approach removes the decoder. Sparsity emerges from competitive responsibility allocation. The objective is likelihood + volume control, not reconstruction + sparsity.

This is "decoder-free" sparse coding: learning sparse representations without explicit reconstruction.

### InfoMax / ICA (Bell & Sejnowski 1995)

Bell & Sejnowski showed that maximizing information transmission leads to independent components. Our variance and decorrelation terms approximate this operationally:

- Maximize variance → entropy proxy under Gaussian assumption
- Minimize correlation → independence proxy for linear dependence

We do not claim to compute mutual information. The connection is that InfoMax-inspired regularization provides the distributional constraints that prevent collapse in EM-based learning.

### Implicit EM (Oursland 2025)

The gradient identity $\partial L/\partial d_j = -r_j$ means:

- Forward pass computes unnormalized likelihoods (energies)
- Normalization yields responsibilities (implicit in gradient)
- Backpropagation delivers responsibility-weighted updates

This paper provides the EM structure. Our contribution is identifying InfoMax regularization as the appropriate volume control—the missing piece that makes unsupervised implicit EM practical.

### Mahalanobis Distance (Oursland 2024)

The interpretation of linear layers as computing Mahalanobis distances to prototypes motivates:

- Weight vectors as prototype directions
- The bias loophole (same direction, different offset)
- Orthogonality regularization to encourage independent features

The weight redundancy term $\|W^\top W - I\|^2$ was predicted in this earlier work as necessary for learning true principal components rather than arbitrary whitening bases.

## Summary Claim

**Decoder-free sparse autoencoders arise naturally from implicit EM when log-sum-exp objectives are combined with explicit volume and redundancy control.**

Reconstruction is one way to impose such control, but not a necessary one. The log-sum-exp structure provides EM dynamics; InfoMax-inspired regularization provides the volume control that prevents collapse. Together, they yield a principled unsupervised objective derived from first principles.

## References

- Bell, A. J., & Sejnowski, T. J. (1995). An information-maximization approach to blind separation and blind deconvolution. Neural Computation.
- Kohonen, T. (1982). Self-organized formation of topologically correct feature maps. Biological Cybernetics.
- LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M., & Huang, F. J. (2006). A tutorial on energy-based learning. In Predicting Structured Data.
- Linsker, R. (1988). Self-organization in a perceptual network. Computer.
- Olshausen, B. A., & Field, D. J. (1996). Emergence of simple-cell receptive field properties by learning a sparse code for natural images. Nature.
- Oursland, A. (2024). Interpreting Neural Networks through Mahalanobis Distance. arXiv:2410.19352.
- Oursland, A. (2025). Gradient Descent as Implicit EM in Distance-Based Neural Models. arXiv:2512.24780.