# NegLogSoftmin: A Type-Preserving Calibration Layer

## The Problem

We are working in a distance convention. Neural outputs are distances from learned prototypes. Lower values indicate better matches. The implicit EM framework requires that these distances be probabilistically calibrated — that exponentiation produces proper responsibilities (posterior probabilities that sum to one).

Raw distances from a linear layer are uncalibrated. A distance of 3 means nothing without knowing the partition function. We need a layer that takes distances in and produces calibrated distances out, preserving the convention.

## Derivation

Start with distances d₁, d₂, ..., d_K. All non-negative (from Softplus).

**Step 1: Convert distances to probabilities.**

Softmin converts distances to probabilities. Lower distance → higher probability.

$$\text{Softmin}(d)_j = \text{Softmax}(-d)_j = \frac{\exp(-d_j)}{\sum_k \exp(-d_k)}$$

This is the responsibility: r_j = Softmin(d)_j. It sums to one. It's the posterior probability that component j explains the input.

Type: distance → probability.

**Step 2: Convert probabilities to logits.**

Taking the log converts probabilities to log-probabilities (logits in the information-theoretic sense).

$$\log r_j = \log \frac{\exp(-d_j)}{\sum_k \exp(-d_k)} = -d_j - \log \sum_k \exp(-d_k)$$

Type: probability → log-probability.

**Step 3: Convert log-probabilities back to distances.**

Negation converts log-probabilities back to distances. Higher log-probability (better match) becomes lower distance (better match). The convention is preserved.

$$-\log r_j = d_j + \log \sum_k \exp(-d_k)$$

Type: log-probability → distance.

## The Result

$$\text{NegLogSoftmin}(d)_j = d_j + \log Z$$

where $Z = \sum_k \exp(-d_k)$ is the partition function.

The full type chain:

```
distance → [Softmin] → probability → [log] → log-probability → [negate] → distance
```

Distance in, distance out. The convention is preserved through every intermediate step.

## What It Does

NegLogSoftmin adds a single scalar — log Z — to every component's distance. This scalar is the same for all components within a sample. Therefore:

- **Relative distances are preserved.** d_i - d_j = y_i - y_j for all i, j.
- **Ranking is preserved.** The closest component stays closest.
- **The shift calibrates.** After the shift, exp(-y_j) = r_j exactly. The output distances are "worth" their probabilistic meaning. Exponentiation directly gives the responsibility without needing a separate normalization step.

The partition function has been absorbed into the representation. Downstream layers receive distances that are already normalized in the probabilistic sense.

## Why It's Needed

Without NegLogSoftmin, the raw distances d are on an arbitrary scale. Two networks could have identical responsibilities (identical mixture assignments) but completely different distance magnitudes. The downstream layer would need to learn to compensate for whatever scale the upstream layer happens to produce.

NegLogSoftmin removes this degree of freedom. It standardizes the distances so that their absolute values carry probabilistic meaning. The downstream layer receives a representation where the numbers mean something fixed: exp(-y_j) is the probability that component j explains the input.

This is what "responsibilities are applied correctly" means. The calibration ensures that when the next layer operates on these values, it operates on quantities with consistent probabilistic semantics.

## Simplified Form

$$y_j = d_j + \log \sum_k \exp(-d_k)$$

This is just the raw distance plus a data-dependent constant. In PyTorch:

```python
y = d + torch.logsumexp(-d, dim=1, keepdim=True)
```

One line. The logsumexp is computed anyway for the LSE loss. The layer is almost free.

## Relationship to Volume Control

NegLogSoftmin calibrates distances. Volume control (LSE + variance + decorrelation) prevents the distances from degenerating. These are complementary:

- Volume control ensures the distances are **healthy** (alive, diverse, competitive).
- NegLogSoftmin ensures the distances are **calibrated** (probabilistically meaningful).

Volume control operates on the raw distances d (the layer input). NegLogSoftmin produces the calibrated distances y (the layer output). The volume control auxiliary loss does not need to be applied to y because the shift is a scalar constant across components — variance and correlation are identical for d and y.

## Gradient Flow

NegLogSoftmin is differentiable. The Jacobian involves responsibilities:

$$\frac{\partial y_i}{\partial d_j} = \mathbb{1}[i = j] - r_j$$

where $r_j = \exp(-d_j) / Z$.

This means:
- The gradient of the supervised loss flows back through NegLogSoftmin to the raw distances.
- The gradient of the auxiliary loss is applied directly to the raw distances.
- Both gradient signals reach W₁ and update the prototype geometry.

The layer participates in implicit EM through its Jacobian. It doesn't just pass values through — it shapes the gradient according to the responsibility structure.

## Summary

NegLogSoftmin is not an activation function or a normalization heuristic. It is a type-preserving calibration layer derived from the probability chain:

```
distance → probability → log-probability → distance
```

Each step has a clear semantic meaning. The composition preserves the distance convention while absorbing the partition function. The result is a representation where distances carry proper probabilistic semantics, ready for downstream consumption.