# Kernels for Implicit EM

## Introduction: Kernels for Implicit EM

This document catalogs a family of **kernels** that give rise to implicit expectation–maximization (EM) dynamics when used in neural models trained with log-sum-exp objectives.

Although the term *objective function* is standard in optimization, it is slightly misleading in this context. The EM behavior of interest does not arise from the global loss as a whole, but from a specific structural component: the **energy-to-likelihood mapping** applied to distance-like representations.

What matters is the local transformation that:

* maps distances or energies to unnormalized likelihoods,
* participates in exponentiation and log-sum-exp normalization,
* induces competition across alternatives,
* produces normalized responsibilities as a consequence.

This local structure is better described as a **kernel** (or **energy kernel**) than as an objective. Each kernel encodes a representational bias: it defines a distance or noise model, determines how deviations are penalized, and controls how evidence accumulates and saturates. The choice of kernel fixes the geometry of inference before any learning dynamics are considered.

Throughout this document:

* *kernel* refers to the local energy or similarity function that converts representations into likelihoods,
* *objective* refers to the global log-sum-exp aggregation constructed from those kernels.

When a kernel is combined with exponentiation, normalization, and gradient-based optimization, implicit EM dynamics are unavoidable. Responsibilities emerge as normalized exponentials, and learning proceeds via responsibility-weighted updates. All kernels enumerated in this document satisfy these conditions and therefore support implicit EM.

---

## General EM-Compatible Form

All cases fit the template:

* Signed or unsigned distance / energy:
  ( d(z, \theta) )

* Unnormalized likelihood:
  ( P(z \mid \theta) = \exp(-d(z, \theta)) )

* Objective:
  ( L = \log \sum_j \exp(-d_j) )

This structure guarantees:

* normalization across alternatives,
* responsibilities as normalized exponentials,
* EM-style gradient dynamics.

---

## 1. Squared Euclidean / Gaussian Kernel

**Energy**
[
d(z) = \frac{(z - \mu)^2}{2\sigma^2}
]

**Interpretation**

* Gaussian noise around a prototype
* L2 / Mahalanobis distance

**Why EM-Compatible**

* Exponentiation yields Gaussian likelihoods
* Log-sum-exp corresponds to marginalization over components
* Classical mixture-of-Gaussians case

**Notes**

* Leads to tanh for signed two-component models
* Strong inductive bias toward smooth, quadratic geometry

---

## 2. Absolute Distance / Laplace Kernel

**Energy**
[
d(z) = \frac{|z - \mu|}{b}
]

**Interpretation**

* Laplace noise
* L1 distance
* Robust to outliers

**Why EM-Compatible**

* Exponentiated L1 distances form Laplace likelihoods
* Normalization induces soft assignment despite non-smooth base distance

**Notes**

* Produces heavier tails than Gaussian
* Encourages sparse, robust deviations

---

## 3. Logistic / Softplus Kernel

**Energy**
[
d(z) = \log!\left(1 + \exp(-\alpha(z - \mu))\right)
]

**Interpretation**

* Logistic noise
* Smooth margin-based distance

**Why EM-Compatible**

* Exponentiation recovers logistic likelihood ratios
* Naturally aligned with binary latent variables

**Notes**

* Closely related to logistic regression and Ising models
* Tanh may arise with different semantics than Gaussian

---

## 4. Student-t / Heavy-Tailed Kernel

**Energy**
[
d(z) = \log!\left(1 + \frac{(z - \mu)^2}{\nu\sigma^2}\right)
]

**Interpretation**

* Student-t noise
* Uncertain or variable scale
* Heavy-tailed deviations

**Why EM-Compatible**

* Exponentiation yields heavy-tailed likelihoods
* Normalization still produces responsibilities

**Notes**

* Extremely robust to outliers
* Encourages cautious, non-committal assignments

---

## 5. Huber Kernel

**Energy**
[
d(z) =
\begin{cases}
\frac{1}{2} z^2 & |z| \le \delta \
\delta(|z| - \frac{1}{2}\delta) & |z| > \delta
\end{cases}
]

**Interpretation**

* Quadratic near zero, linear in the tails
* Compromise between Gaussian and Laplace

**Why EM-Compatible**

* Piecewise energy still exponentiates cleanly
* Normalization induces competition across components

**Notes**

* Allows local precision with global robustness
* Common in robust statistics

---

## 6. Pseudo-Huber Kernel

**Energy**
[
d(z) = \delta^2\left(\sqrt{1 + (z/\delta)^2} - 1\right)
]

**Interpretation**

* Smooth approximation to Huber
* Differentiable everywhere

**Why EM-Compatible**

* Avoids non-differentiability while preserving robustness
* Well-behaved gradients under EM dynamics

**Notes**

* Often preferable in deep models for smoothness

---

## 7. Exponential Distance Kernel

**Energy**
[
d(z) = \exp(\alpha |z|)
]

**Interpretation**

* Extremely sharp penalty
* High sensitivity to deviations

**Why EM-Compatible**

* Still yields valid likelihoods under exponentiation
* Normalization enforces competition

**Notes**

* Very aggressive assignment behavior
* Rarely stable without temperature control

---

## 8. Cosine / Angular Kernel

**Energy**
[
d(x, \mu) = 1 - \frac{x^\top \mu}{|x||\mu|}
]

**Interpretation**

* Angular distance
* Directional similarity

**Why EM-Compatible**

* Exponentiated cosine similarity used widely in attention
* Softmax normalization induces responsibilities

**Notes**

* Scale-invariant
* Common in metric learning and representation learning

---

## 9. Quadratic Form / General Mahalanobis Kernel

**Energy**
[
d(x) = (x - \mu)^\top \Sigma^{-1}(x - \mu)
]

**Interpretation**

* Full covariance Gaussian
* Learned metric geometry

**Why EM-Compatible**

* Canonical mixture-model distance
* Responsibilities correspond to classical EM

**Notes**

* Matches the “distance-based neural representation” view directly

---

## 10. Energy Networks / Learned Energy Kernels

**Energy**
[
d(x) = f_\theta(x)
]

**Interpretation**

* Learned energy surface
* No explicit noise model

**Why EM-Compatible**

* As long as energies are exponentiated and normalized,
  responsibilities arise regardless of interpretability

**Notes**

* Most general case
* Semantics depend entirely on training dynamics

---

## Common Requirement Across All Kernels

A kernel supports implicit EM if and only if:

* it produces scalar energies,
* energies are exponentiated,
* likelihoods are normalized across alternatives,
* optimization is gradient-based.

The specific functional form determines:

* robustness,
* saturation behavior,
* inductive bias,
* and the resulting activation shape.

The EM mechanism itself is invariant.
