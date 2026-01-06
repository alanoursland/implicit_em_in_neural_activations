# InfoMax as the Objective for Implicit EM

## The Separation

Implicit EM has two parts:

1. **Mechanism:** How prototypes update (responsibility-weighted gradients)
2. **Objective:** What prototypes learn (task-dependent)

The softmax/LSE structure provides the mechanism. It doesn't specify the objective.

In supervised learning, cross-entropy provides the objective: match labels. The EM mechanism implements it via soft competition among classes.

In unsupervised learning, what's the objective?

## InfoMax as the Answer

InfoMax: maximize information in the output, minimize redundancy.

```
L = -Σⱼ H(zⱼ) + λ TC(z)
```

First term: each output should be informative (high marginal entropy)
Second term: outputs should be independent (low total correlation)

Equivalently: maximize joint entropy H(z₁, ..., zₖ).

## Why InfoMax is Right

**Prevents collapse:**

A dead prototype (always large distance) has zero variance. H(zⱼ) = 0. Hurts the objective.

A dominant prototype (always wins) makes other outputs deterministic given it. TC increases. Hurts the objective.

InfoMax requires all prototypes to be active and informative.

**Prevents redundancy:**

Two prototypes measuring the same thing have correlated outputs. TC increases. Directly penalized.

InfoMax forces each prototype to detect something unique.

**Maximizes expressiveness:**

K independent binary partitions create 2^K regions. This is maximum combinatorial expressiveness for K hyperplanes.

Correlated partitions create fewer regions. Information is wasted.

InfoMax gives maximum regions, maximum expressiveness.

## The Architecture

```
Input: x ∈ ℝⁿ
Linear: z = Wx + b ∈ ℝᴷ
Activation: r = softmax(-z) (interpreting z as distances)
Loss: L_InfoMax = -Σⱼ H(zⱼ) + λ TC(z)
```

The InfoMax loss operates on pre-softmax z. The softmax produces responsibilities for downstream use or for understanding the learned structure.

Gradients flow through softmax to W. The chain rule delivers responsibility-weighted updates. This is the EM mechanism.

## What Each Part Contributes

| Component | Role |
|-----------|------|
| Linear layer | Computes distances to prototypes |
| Softmax | Converts distances to responsibilities, provides EM gradient structure |
| InfoMax on z | Objective: diverse, informative prototypes |

The mechanism (softmax) and objective (InfoMax) are orthogonal. You could swap InfoMax for another unsupervised objective. You could swap softmax for another competitive activation. They're separable design choices.

## Connection to Classical EM

Classical GMM-EM:
- Objective: log marginal likelihood log Σₖ P(x|k)
- Mechanism: E-step computes responsibilities, M-step updates prototypes

Implicit EM with InfoMax:
- Objective: InfoMax on distances
- Mechanism: softmax computes responsibilities, gradient descent updates prototypes

The objectives differ. GMM-EM asks: can the mixture explain the data (density estimation)? InfoMax asks: do the prototypes capture independent features (representation learning)?

Both use responsibility-weighted updates. The destination differs.

## Connection to Volume Control

Previous notes identified collapse as a risk for competitive activations. GMMs prevent collapse with the log-determinant (volume penalty). What prevents collapse here?

InfoMax is the volume control.

The marginal entropy term -Σⱼ H(zⱼ) requires each output to have variance. Dead prototypes have zero entropy. Penalized.

The total correlation term requires outputs to be independent. A dominant prototype creates dependence. Penalized.

InfoMax naturally prevents both collapse modes:
- Winner-take-all: blocked by TC term
- Dead units: blocked by marginal entropy term

No additional regularization needed. The objective handles it.

## Practical Computation

Exact entropy and total correlation require density estimation. Expensive.

Approximations:

**Gaussian assumption:**
If z is approximately Gaussian, H(z) = ½ log det(Cov(z)) + const.
TC reduces to correlation structure.
Loss becomes: -log det(C) + λ ||C - diag(C)||²
Where C = Cov(z) estimated from batch.

**Decorrelation proxy:**
If we only care about linear dependence:
L = ||Cᵀ C - I||²
Where C is column-normalized z over the batch.
This is Barlow Twins. Tractable. Catches most redundancy.

**HSIC:**
For nonlinear dependence:
L = Σⱼ<ₖ HSIC(zⱼ, zₖ)
More expensive. More thorough.

Start with Gaussian/decorrelation. Move to HSIC if needed.

## Batch Dependence

InfoMax requires batch statistics. H(zⱼ) and TC(z) are properties of the distribution, estimated from the batch.

This is unavoidable. Collapse prevention requires knowing: "is this prototype used by other inputs?" A single sample can't answer that.

Like BatchNorm, InfoMax couples samples within a batch. Unlike BatchNorm, it has a principled information-theoretic motivation.

## The Full Picture

For unsupervised training of a single layer:

```
Forward:
  z = Wx + b
  r = softmax(-z)  [optional, for interpretation]

Loss:
  L = -Σⱼ H(zⱼ) + λ TC(z)
  [approximated via batch covariance]

Backward:
  Gradients flow through softmax (if used) or directly to z
  W receives responsibility-weighted updates
  This is implicit EM
```

The layer learns K prototypes that:
- Cover different parts of the input space (EM dynamics)
- Detect independent features (InfoMax objective)
- Are all utilized (entropy term)
- Are not redundant (TC term)

## What Remains

1. **Experimental validation:** Does this actually produce non-redundant hyperplanes on real data?

2. **Stacking:** If layer 1 has InfoMax, what does layer 2 see? Does InfoMax compose?

3. **Activation choice:** We derived this with softmax. Does it work with other competitive activations?

4. **Supervised extension:** How does a supervised signal interact with InfoMax? Auxiliary loss? Constraint?

5. **Approximation quality:** Is Gaussian/decorrelation good enough, or does nonlinear dependence matter?

---

## Summary

InfoMax is the objective. Implicit EM is the mechanism.

Softmax provides responsibility-weighted updates. InfoMax tells those updates what to achieve: independent, informative features.

This resolves the question of unsupervised layer-wise training. Each layer maximizes information about its input while ensuring no redundancy. The EM dynamics implement this efficiently.