# Deriving Decoder-Free Sparse Autoencoders from Implicit EM

## Introduction

This document derives a neural architecture from first principles using the theory of implicit expectation-maximization (EM) in gradient descent. The result is a model equivalent to decoder-free sparse autoencoders and closely related to energy-based models, but arrived at through a different path: the algebraic structure of log-sum-exp objectives.

## Starting Point: Gradient Descent as Implicit EM

Oursland (2025) establishes a key identity. For any objective with log-sum-exp structure:

```
L = log Σⱼ exp(-dⱼ)
```

The gradient with respect to each distance is exactly the negative responsibility:

```
∂L/∂dⱼ = -rⱼ where rⱼ = exp(-dⱼ) / Σₖ exp(-dₖ)
```

This is algebraic, not approximate. The immediate consequence: gradient descent on such objectives performs EM implicitly. Responsibilities are not computed as auxiliary variables—they are the gradients.

## The Question: What Objective for Unsupervised Learning?

The identity holds at the interface between model outputs and loss. If we want EM structure throughout training, we need a log-sum-exp loss.

For supervised learning, cross-entropy provides this:
```
L = -log P(y|x) = -zᵧ + log Σₖ exp(zₖ)
```

For unsupervised learning, what is the objective?

The natural choice is the marginal likelihood:
```
L = -log Σⱼ exp(-dⱼ)
```

This says: at least one component should explain each input. Maximizing this encourages prototypes to cover the data.

## The Collapse Problem

Pure log-sum-exp has a failure mode. One component can claim all inputs. Its distance shrinks, its responsibility approaches 1, other components receive no gradient, they die.

In Gaussian mixture models, the log-determinant prevents this:
```
log P(x|k) ∝ -½(x-μₖ)ᵀΣₖ⁻¹(x-μₖ) - ½ log det(Σₖ)
```

The log-determinant penalizes small covariance. A component cannot shrink to a point.

Neural networks lack this term. The paper notes this explicitly: collapse is a risk because volume control is missing.

## InfoMax as Volume Control

What prevents collapse in unsupervised neural learning?

We propose InfoMax regularization. InfoMax (Linsker 1988, Bell & Sejnowski 1995) maximizes information by:
1. Maximizing entropy of each output (variance proxy)
2. Minimizing redundancy between outputs (decorrelation)

**Entropy term:** -Σⱼ log(Var(aⱼ))

This prevents collapse. A dead component has zero variance. The penalty is infinite.

**Decorrelation term:** ||Corr(A) - I||²

This prevents redundancy. If two components encode the same thing, they're correlated. The penalty grows.

Together, these serve the role of the log-determinant: they enforce that components maintain volume and diversity.

## The Complete Objective

Combining log-sum-exp (EM structure) with InfoMax (volume control):

```
L = -log Σⱼ exp(-aⱼ) - λ_var Σⱼ log(Var(aⱼ) + ε) + λ_tc ||Corr(A) - I||²
```

**Term 1:** Marginal likelihood. At least one prototype should be close to each input. Provides EM gradient structure.

**Term 2:** Entropy. All prototypes must be active. Prevents collapse.

**Term 3:** Decorrelation. Prototypes must be different. Prevents redundancy.

## The Sign-Flip: A Min-Max Game

There is a crucial sign difference between standard clustering and our model.

**Standard clustering EM:**
```
max log Σⱼ exp(zⱼ)  → pull prototypes toward data
```

**Our model:**
```
min log Σⱼ exp(zⱼ)  → push prototypes apart
```

This inverts the role of the LSE term. Minimizing LSE penalizes prototypes that are too close to inputs relative to others. It's a repulsive force.

**InfoMax is the attractive force.**

The entropy term -Σⱼ log(Var(aⱼ)) rewards prototypes that respond strongly to inputs. To have high variance, a prototype must be close to some data. InfoMax pulls prototypes toward the data.

**The model is a min-max game:**

| Term | Force | Effect |
|------|-------|--------|
| InfoMax (entropy) | Attractive | Pull prototypes toward data |
| LSE | Repulsive | Push prototypes apart (distribute responsibility) |

This implements competitive learning:
- Every prototype tries to get close to data (InfoMax)
- LSE taxes prototypes proportional to their responsibility
- The equilibrium: prototypes spread out to cover the data

**Connection to prior work:**

This is differentiable vector quantization. Similar dynamics appear in:
- Self-Organizing Maps (Kohonen)
- Competitive learning networks
- Vector quantization

The difference: our formulation is fully differentiable and probabilistic. No hard winner selection. Soft responsibilities via the implicit softmax in the LSE gradient.

## The Architecture

The minimal architecture is:

```
z = Wx + b
a = relu(z)
L = logsumexp(-a) + infomax(a)
```

**Linear layer:** Computes distances to prototypes. Each row of W is a prototype direction.

**ReLU:** Ensures non-negative activations. Introduces sparsity. Low activation = far from prototype.

**LogSumExp loss:** Provides EM gradient structure. ∂L/∂aⱼ = -rⱼ.

**InfoMax terms:** Volume control. Prevent collapse and redundancy.

No decoder. No reconstruction. The objective is implicit likelihood with information-theoretic regularization.

## Connection to Prior Work

### Energy-Based Models (LeCun et al. 2006)

EBMs define probability via energy:
```
P(x) ∝ exp(-E(x))
```

Our logsumexp term is:
```
L = -log Σⱼ exp(-aⱼ) = -log Σⱼ exp(-Eⱼ(x))
```

This is the negative log marginal likelihood under a mixture of energy functions. Each prototype j defines an energy Eⱼ(x) = aⱼ = relu(wⱼᵀx + bⱼ).

The InfoMax terms are regularization on the energy landscape—ensuring diverse, well-spread energies.

### Sparse Coding (Olshausen & Field 1996)

Sparse coding learns:
```
min ||x - Da||² + λ||a||₁
```

Decoder D, sparse code a. Reconstruction + sparsity.

Our approach removes the decoder. Sparsity emerges from ReLU. The objective is likelihood + information preservation, not reconstruction.

This is "decoder-free" sparse coding: learn sparse representations without explicit reconstruction.

### InfoMax / ICA (Bell & Sejnowski 1995)

Bell & Sejnowski showed that maximizing information transmission leads to independent components:
```
max I(X; A) = max H(A) - H(A|X)
```

Our entropy and decorrelation terms approximate this:
- Maximize variance (entropy proxy under Gaussian assumption)
- Minimize correlation (independence proxy for linear dependence)

The connection: InfoMax is the right regularizer for EM-based learning because it enforces the distributional properties that prevent collapse.

### Implicit EM (Oursland 2025)

The gradient identity ∂L/∂dⱼ = -rⱼ means:
- Forward pass computes unnormalized likelihoods
- Normalization yields responsibilities
- Backpropagation delivers responsibility-weighted updates

This paper provides the EM structure. Our contribution is identifying InfoMax as the appropriate volume control.

## Do We Need Explicit Decorrelation and Weight Regularization?

Empirically, we found:

**Without InfoMax (pure logsumexp):** Collapse occurs. One prototype dominates.

**With entropy term only:** Collapse prevented. All units active. But redundancy possible.

**With entropy + decorrelation:** Non-redundant outputs. But weights can still be redundant (bias loophole with ReLU).

**With weight redundancy penalty:** Forces diverse weight directions. Closes all loopholes.

However, the necessity depends on architecture:

| Configuration | Needs λ_tc | Needs λ_wr |
|---------------|-----------|-----------|
| Linear → ReLU → LogSumExp | Maybe | Yes |
| Linear → Softmax → LogSumExp | Less | Less |

When softmax appears in the forward pass, competition provides implicit regularization. Weight vectors naturally differentiate to win different responsibilities.

When using only logsumexp loss (softmax implicit in gradient), explicit regularization is needed to prevent degenerate solutions.

**Recommendation:** Start with full InfoMax regularization (λ_tc > 0, λ_wr > 0). Ablate to find minimal necessary terms for your architecture.

## Theoretical Summary

**Claim:** Decoder-free sparse autoencoders can be derived from implicit EM theory.

**Derivation:**
1. Implicit EM requires logsumexp loss structure
2. Logsumexp alone collapses (missing volume control)
3. InfoMax provides volume control (entropy prevents death, decorrelation prevents redundancy)
4. Combined objective: logsumexp + InfoMax

**Result:**
```
z = Wx + b
a = relu(z)
L = -log Σⱼ exp(-aⱼ) + λ_var(-Σⱼ log Var(aⱼ)) + λ_tc ||Corr(A) - I||² + λ_wr ||WᵀW - I||²
```

**Interpretation:**
- LogSumExp: MLE under mixture model, provides EM gradient structure
- Entropy: Prevents collapse (analogous to log-determinant)
- Decorrelation: Prevents redundant representations
- Weight redundancy: Prevents redundant prototypes

## Relation to the Original Paper

The paper "Gradient Descent as Implicit EM" establishes that EM structure exists at the output layer for logsumexp objectives. It notes that internal layers do not have this structure and that collapse is a risk without volume control.

Our contribution:
1. **Locate the missing piece:** InfoMax serves as volume control
2. **Derive the architecture:** Linear → Activation → LogSumExp + InfoMax
3. **Connect to prior work:** This is decoder-free sparse coding / EBM arrived at from EM principles
4. **Empirically validate:** Document failure modes and solutions

The paper asks: what are neural networks doing? Answer: implicit EM at the output.

We ask: how do we build a layer that does full implicit EM? Answer: logsumexp loss with InfoMax regularization.

## References

- Bell, A. J., & Sejnowski, T. J. (1995). An information-maximization approach to blind separation and blind deconvolution. Neural Computation.
- LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M., & Huang, F. J. (2006). A tutorial on energy-based learning. In Predicting Structured Data.
- Linsker, R. (1988). Self-organization in a perceptual network. Computer.
- Olshausen, B. A., & Field, D. J. (1996). Emergence of simple-cell receptive field properties by learning a sparse code for natural images. Nature.
- Oursland, A. (2025). Gradient Descent as Implicit EM in Distance-Based Neural Models. arXiv:2512.24780.