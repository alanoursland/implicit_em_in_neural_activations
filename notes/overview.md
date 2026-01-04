# Implicit EM in Neural Activations — Overview Notes

## Core Claim

Neural activation functions are not primitive nonlinearities.
They are posterior summaries of implicit expectation–maximization (EM) induced by the training objective.

Given:

* distance- or energy-based representations (e.g. Mahalanobis coordinates from linear layers),
* exponentiation and normalization via log-sum-exp objectives,
* and gradient-based optimization,

EM-style inference is unavoidable. Responsibilities arise as gradients. Activations arise as posterior statistics of those responsibilities.

This paper derives activation functions as *consequences* of objective geometry, not as design choices.

---

## Relationship to Prior Work

The companion paper establishes:

> For any objective of the form
> ( L = \log \sum_j \exp(-d_j) ),
> the gradient with respect to each distance is the negative responsibility.

Thus:

* forward pass = implicit E-step,
* backward pass = implicit M-step,
* responsibilities are never auxiliary variables; they are gradients.

This paper closes the loop by showing:

> Activations are posterior expectations computed from those same responsibilities.

---

## Unifying Setup

Assumptions fixed throughout:

* A linear layer produces signed, distance-like coordinates ( z = Wx + b ).
* These coordinates are interpreted geometrically (Mahalanobis-style).
* Training uses objectives with log-sum-exp structure over energies.
* No explicit latent variables or EM algorithm are introduced.

Under these conditions:

* normalization induces competition,
* competition induces responsibilities,
* responsibilities define posterior distributions over latent hypotheses.

---

## Key Perspective Shift

Traditional view:

* activation functions are architectural heuristics,
* chosen for optimization convenience or historical reasons.

This paper’s view:

* activation functions are posterior summaries of implicit latent-variable inference,
* fully determined by the assumed distance / noise model in the objective,
* independent of architectural idiosyncrasies.

The nonlinearity is not “inserted.”
It is *derived*.

---

## General Derivation Pattern

For each activation:

1. Specify a latent hypothesis structure (e.g. signed causes).
2. Specify an energy or distance model ( d(z, s) ).
3. Form the normalized exponential via log-sum-exp.
4. Compute posterior responsibilities.
5. Take a posterior statistic (typically the mean).

The resulting function is the activation.

Different objectives ⇒ different activations.
Same EM mechanism throughout.

---

## Canonical Example: Tanh

Under a symmetric two-component Gaussian distance model:

* latent sign ( s \in {\pm 1} ),
* energies ( d_\pm(z) = \frac{(z \mp m)^2}{2\sigma^2} ),

the posterior expectation is:
[
\mathbb{E}[s \mid z] = \tanh!\left(\frac{m}{\sigma^2} z\right).
]

Thus:

* tanh is the posterior mean of a signed latent variable,
* its scale parameter corresponds to separation / variance,
* it preserves soft responsibilities everywhere (no dead regions).

---

## Beyond Gaussian Objectives

Changing the distance / noise model changes the activation:

* Laplace distances ⇒ soft-sign–like nonlinearities
* Heavy-tailed distances (e.g. Student-t) ⇒ slow-saturating odd functions
* Robust losses ⇒ hybrid linear–saturating behaviors

In all cases:

* responsibilities still arise as gradients,
* EM dynamics remain implicit,
* the activation is a posterior statistic, not a heuristic.

---

## What This Paper Does *Not* Claim

* No new architectures are proposed.
* No empirical performance claims are made.
* No explicit probabilistic models are added to networks.
* No variational approximations are required.

The contribution is explanatory and structural.

---

## Takeaway

EM is not something neural networks approximate accidentally.
It is enforced by objective geometry.

Activation functions are not arbitrary nonlinearities.
They are posterior inference states.

Once distances, exponentiation, and normalization are present, both EM and the resulting activations are inevitable.
