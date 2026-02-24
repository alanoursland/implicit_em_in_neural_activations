# Softplus as Kernel Choice

## What a Kernel Is in This Context

The implicit EM framework requires energies — scalar values assigned to each component for each input. The kernel is the function that converts raw projections (pre-activations) into those energies. It determines the noise model, the distance geometry, and the inductive bias of the mixture.

The kernel is not the activation function. It is the energy assignment step in the pipeline:

```
z = Wx + b          → signed projection (scalar)
d = kernel(z)       → energy (non-negative distance)
```

Different kernels encode different assumptions about what "distance from a prototype" means. The choice of kernel is a modeling decision with consequences for the mixture geometry.

## The Candidates

From kernels_for_implicit_em.md, the main options for scalar projections are:

### Squared Euclidean (Gaussian kernel)

```
d(z) = z² / 2σ²
```

Gaussian noise model. Quadratic penalty. Small deviations cheap, large deviations expensive. Symmetric around zero. The classical GMM distance.

### Absolute Distance (Laplace kernel)

```
d(z) = |z| / b
```

Laplace noise model. Linear penalty. Robust to outliers — large deviations penalized linearly, not quadratically. Symmetric around zero. Non-differentiable at zero.

### Softplus (Logistic kernel)

```
d(z) = log(1 + exp(z))
```

Logistic noise model. Smooth, always positive. Approximately linear for z >> 0. Approximately exp(z) for z << 0. Differentiable everywhere.

### ReLU (Hard threshold)

```
d(z) = max(0, z)
```

Not a proper kernel. Hard gating at zero. Entire half-space mapped to distance zero. Non-differentiable at zero. Discussed separately because of its prevalence.

## Why Softplus

### Non-negative everywhere

Distances must be non-negative. A distance of -3 is meaningless. Softplus is always positive: Softplus(z) > 0 for all z. The output is a proper distance regardless of the input.

Gaussian kernel (z²) also achieves this. Absolute distance (|z|) also achieves this. ReLU achieves this but with a degenerate zero region.

### Smooth everywhere

The gradient of Softplus is the sigmoid: Softplus'(z) = σ(z) = 1/(1 + exp(-z)). This is smooth, bounded between 0 and 1, and never zero.

Gaussian kernel is smooth. Absolute distance is not differentiable at z = 0. ReLU is not differentiable at z = 0 and has zero gradient for z < 0.

Smoothness matters for EM dynamics. The responsibility-weighted gradient must flow to all components. A non-differentiable point or a zero-gradient region blocks gradient flow and breaks the implicit M-step.

### No flat region

This is the critical distinction from ReLU. ReLU maps the entire half-space z ≤ 0 to distance zero. Every input in that half-space is "perfectly matched" to the component. The prototype is not a point or a surface — it is an infinite region.

For mixture modeling, this is pathological. A component that claims an infinite region at zero cost has no incentive to specialize. Its responsibility is determined entirely by inputs in the z > 0 region. Inputs in the z ≤ 0 region contribute nothing to differentiation.

Softplus has no flat region. Every input has a unique, positive distance to every component. Moving the input slightly always changes the distance. The prototype surface (minimum distance) is a smooth manifold at z = -∞, approached but never reached. In practice, inputs near z = 0 have distance ≈ log(2) ≈ 0.69 — small but nonzero.

### Asymptotic behavior

For large positive z: Softplus(z) ≈ z. Linear penalty. Similar to Laplace kernel. Distant inputs are penalized linearly, not quadratically. This provides robustness — extremely distant inputs don't dominate the loss with huge squared distances.

For large negative z: Softplus(z) ≈ exp(z) ≈ 0. The distance approaches zero exponentially. Inputs deep in the "matching" region have near-zero distance. The component is confident about these inputs.

This asymmetry is natural for the distance interpretation. Close matches → near-zero distance (high confidence). Distant mismatches → linear penalty (not catastrophic). The transition is smooth.

### Gradient behavior

Softplus'(z) = σ(z).

For z >> 0 (far from prototype): σ(z) ≈ 1. Full gradient. The component receives strong signal to adjust.

For z << 0 (close to prototype): σ(z) ≈ 0. Small gradient. The component is already well-matched; little adjustment needed.

For z ≈ 0 (boundary region): σ(z) ≈ 0.5. Moderate gradient. The interesting region where assignment is uncertain.

This gradient profile is natural for mixture modeling. Components get strong signal from inputs they explain poorly (need to adjust) and weak signal from inputs they explain well (already good). The gradient is self-regulating.

## Why Not ReLU

Paper 2 used ReLU. It worked for the unsupervised model on MNIST. So why change?

### ReLU breaks EM at the component level

From why_relu_breaks_em.md: ReLU implements hard assignment. For z ≤ 0, the component outputs distance zero and receives zero gradient through the activation. From the EM perspective, this is a dead hypothesis for that input — permanently assigned, never revised.

In the unsupervised model, this was partially masked by the volume control terms operating across the batch. The variance penalty ensures each component has nonzero variance, which means each component must have some inputs in its z > 0 region. But the EM dynamics are still broken for individual inputs in the z ≤ 0 region.

### The half-space problem

With ReLU, each component's "prototype" is a half-space boundary. The zero-distance region is not a point or a small neighborhood — it is half of input space (projected onto the component's direction). Multiple inputs can have identical distance (zero) to the same component, providing no information about which input the component matches better.

With Softplus, the minimum distance approaches zero but every input has a distinct distance. The component can distinguish between "very close" and "extremely close." This matters for responsibility computation — softer, more informative assignments.

### Compatibility with NegLogSoftmin

NegLogSoftmin calibrates distances so that exp(-yⱼ) = rⱼ. With ReLU distances, many distances are exactly zero. exp(-0) = 1 for all of them. Components with zero distance have equal unnormalized likelihood regardless of how well they actually match. The calibration has nothing meaningful to work with.

With Softplus distances, every component has a distinct, positive distance. The calibration produces meaningful responsibilities that reflect genuine differences in fit.

### Paper 2's results still hold

The implicit EM identity ∂L_LSE/∂dⱼ = rⱼ holds for any differentiable distance. Softplus is differentiable everywhere. The theorem verification experiment should replicate identically.

The ablation predictions (LSE collapses, variance prevents death, decorrelation prevents redundancy) depend on the LSE + InfoMax structure, not on the specific kernel. Softplus changes the geometry of the mixture but not the dynamics of competition and volume control.

## Why Not Gaussian Kernel (z²)

The Gaussian kernel d(z) = z²/2σ² is the classical choice for mixture models. Why not use it?

### Quadratic penalty is aggressive

Large deviations are penalized quadratically. An input at z = 10 has distance 50 (with σ = 1). Its likelihood exp(-50) ≈ 0 and its responsibility is negligible. The component effectively ignores all distant inputs.

This is fine for classical GMMs where the number of components is chosen to cover the data. In a neural network where components must also produce useful representations, aggressive truncation may be harmful. Components only learn from nearby inputs, making them very local.

### Scale sensitivity

The Gaussian kernel has a scale parameter σ that must be set or learned. Too small: components see only their immediate neighborhood. Too large: components are insensitive to distance. The scale interacts with the weight magnitudes and the input distribution.

Softplus has no explicit scale parameter. The effective scale is absorbed into the weight magnitudes, which are learned. One fewer thing to tune.

### Symmetry

d(z) = z² is symmetric: deviations in both directions are penalized equally. This is natural for Gaussian noise but means the kernel treats z = +5 and z = -5 identically. The signed information in the projection is discarded.

Softplus is asymmetric. Positive z (far from prototype in the "wrong" direction) is penalized linearly. Negative z (close to prototype) approaches zero distance. The sign carries information about which side of the prototype the input falls on.

## Why Not Absolute Distance (|z|)

### Non-differentiable at zero

|z| has a kink at z = 0. The gradient is undefined at the prototype surface. In practice, subgradients or smoothed approximations are used, but this introduces a discontinuity in the learning dynamics exactly where it matters most — at the decision boundary.

Softplus is the smooth approximation to ReLU, which is itself the positive half of |z|. Using Softplus directly avoids the non-differentiability while preserving the linear tail behavior.

### Otherwise attractive

The Laplace kernel has good robustness properties and a natural interpretation. If differentiability weren't an issue, it would be a strong candidate. The pseudo-Huber kernel (smooth approximation to Huber, which interpolates between quadratic and linear) is another option in this family.

## Summary

| Property | ReLU | Softplus | Gaussian (z²) | Laplace (\|z\|) |
|----------|:----:|:--------:|:--------------:|:----------------:|
| Non-negative | ✓ | ✓ | ✓ | ✓ |
| Smooth | ✗ | ✓ | ✓ | ✗ |
| No flat region | ✗ | ✓ | ✓ | ✓ |
| Bounded gradient | ✗ (binary) | ✓ (0,1) | ✗ (grows with z) | ✗ (constant) |
| Robust to outliers | ✗ | ✓ (linear tail) | ✗ (quadratic) | ✓ (linear) |
| Scale-free | ✓ | ✓ | ✗ (needs σ) | ✗ (needs b) |
| EM-compatible | ✗ | ✓ | ✓ | ✓ (with smoothing) |

Softplus is the choice that satisfies all requirements: non-negative, smooth, no flat region, bounded gradient, robust tails, no scale parameter, EM-compatible. It is the logistic kernel — the smooth interpolation between exponential (near prototype) and linear (far from prototype) distance behavior.

For our supervised model, this is the right default. If the activation geometry matters (an open question from Paper 2's future directions), systematic comparison across kernels is a natural follow-up. For now, Softplus is the principled choice that avoids ReLU's known pathologies without introducing unnecessary complexity.