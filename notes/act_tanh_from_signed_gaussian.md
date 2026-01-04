## Signed Gaussian to tanh

### Overview

This note derives the **tanh activation** from a **signed Gaussian kernel**, interpreted as a symmetric Mahalanobis distance model with an explicit latent sign variable. The derivation follows a consistent pattern intended to be reused across activation proofs.

The key idea is that tanh is not an arbitrary nonlinearity. It is the **posterior mean** induced by a signed Gaussian distance kernel under log-sum-exp normalization, i.e. implicit EM.

---

## Gaussian Kernel and Mahalanobis Distance

### Gaussian likelihood

Consider a one-dimensional Gaussian:
[
p(z \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}
\exp!\left(-\frac{(z-\mu)^2}{2\sigma^2}\right).
]

The negative log-likelihood (up to constants) is:
[
d(z;\mu) = \frac{(z-\mu)^2}{2\sigma^2}.
]

This is a **squared Mahalanobis distance** in 1D.

### Mahalanobis interpretation of a linear layer

Let
[
z = w^\top x + b.
]

This can be interpreted as a **projection onto a learned principal direction** (w), with offset (b). In the Gaussian view:

* (w) defines a principal axis,
* (b) defines the mean offset along that axis,
* (z) measures signed deviation from the mean,
* squared deviation corresponds to Gaussian energy.

In higher dimensions, this generalizes to:
[
d(x) = (x-\mu)^\top \Sigma^{-1} (x-\mu),
]
with the linear layer representing a low-rank or diagonalized Mahalanobis metric.

---

## Absolute Value and ReLU as Unsigned Distance

In standard distance-based constructions, one often forms an **unsigned distance**:
[
|z| = |w^\top x + b|.
]

This discards orientation and yields a symmetric deviation measure. In ReLU networks, this is implemented implicitly via:
[
|z| = \mathrm{ReLU}(z) + \mathrm{ReLU}(-z),
]
which decomposes signed distance into two half-space detectors.

While this preserves distance magnitude, it destroys sign information and enforces **hard assignment**: one side is inactive with zero gradient. From the EM perspective, this corresponds to collapsing responsibility mass to a single hypothesis.

To preserve soft competition and avoid dead regions, the absolute value (and by extension ReLU) must be removed.

---

## Signed Gaussian Kernel

### Latent sign variable

Introduce a latent variable:
[
s \in {+1,-1},
]
representing the side of the hyperplane.

Assume a symmetric Gaussian model:
[
p(z \mid s) = \mathcal N(s,m,\sigma^2),
]
with uniform prior (p(s=\pm1)=\tfrac12).

### Energy form

The corresponding energies are:
[
d_s(z) = \frac{(z - s m)^2}{2\sigma^2}.
]

This defines a **signed Mahalanobis distance kernel**:

* two symmetric Gaussian components,
* equal variance,
* separated by (2m),
* centered on opposite sides of the hyperplane.

---

## Log-Sum-Exp and Responsibilities

Form the log-sum-exp objective over (s):
[
L(z) = \log \sum_{s\in{\pm1}} \exp!\big(-d_s(z)\big).
]

Normalization induces responsibilities:
[
r_s(z) = \frac{\exp(-d_s(z))}{\exp(-d_{+}(z))+\exp(-d_{-}(z))}.
]

This is the implicit E-step. Responsibilities are well-defined everywhere and never hard-zero.

---

## Activation as Posterior Expectation

Define the activation as the posterior mean of the latent sign:
[
a(z) = \mathbb E[s\mid z] = r_{+}(z) - r_{-}(z).
]

Compute the energy difference:
[
d_{+}(z) - d_{-}(z)
= \frac{(z-m)^2 - (z+m)^2}{2\sigma^2}
= -\frac{2m}{\sigma^2}z.
]

Thus:
[
\frac{r_{+}(z)}{r_{-}(z)} = \exp!\left(\frac{2m}{\sigma^2}z\right),
\qquad
r_{+}(z) = \sigma!\left(\frac{2m}{\sigma^2}z\right),
]
where (\sigma(\cdot)) is the logistic sigmoid.

Therefore:
[
a(z) = 2r_{+}(z)-1
= \tanh!\left(\frac{m}{\sigma^2}z\right).
]

---

## Result

[
\boxed{
\text{Signed Gaussian kernel}
;\Longrightarrow;
\text{tanh activation}
}
]

The scale parameter (\alpha = m/\sigma^2) is the **inverse temperature** of the EM competition:

* larger (\alpha) ⇒ sharper assignments,
* smaller (\alpha) ⇒ softer assignments.

---

## Interpretation

* The kernel is **Gaussian** (quadratic energy).
* The distance is **Mahalanobis** (learned linear projection).
* The sign is a **latent hypothesis**, not a hard threshold.
* The activation is a **posterior statistic**, not a heuristic nonlinearity.

Tanh emerges inevitably once:

1. signed distances are preserved,
2. energies are exponentiated,
3. normalization induces competition.

This template generalizes directly to other kernels by replacing the Gaussian energy while keeping the same EM structure.
