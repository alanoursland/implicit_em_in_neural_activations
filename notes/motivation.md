# Motivation

The goal of this work is to understand how **expectation–maximization (EM) structure can be realized at the level of individual neural layers**, rather than only at the level of the global objective.

Under a distance-based interpretation of neural representations, the outputs of linear layers are best understood as deviations from learned reference structures—projections onto principal directions or low-rank Mahalanobis coordinates. When such distances are trained under log-sum-exp objectives, EM-like behavior arises globally through exponentiation and normalization. This naturally raises a more local question: can the same mechanism be made explicit *within* layers, so that each layer participates in soft assignment rather than relying on ad hoc nonlinearities?

A natural starting point is to consider softmin, since it converts distances into normalized likelihoods. This led to the initial idea:

> Replace the activation function with a softmin, so that each linear layer directly participates in EM.

This framing was suggestive, but ultimately incomplete.

---

## Why Direct Softmin Is Insufficient

Softmin performs a very specific operation: it **normalizes across a set of competing alternatives**. It takes a collection of energies and produces responsibilities. On its own, it does not define a meaningful transformation of a single scalar.

Applying softmin directly to the output of an individual linear unit—particularly after forming an absolute value—leads to several conceptual problems:

* A single scalar does not define a hypothesis set, so there is no meaningful competition.
* Absolute values remove orientation, eliminating information needed to distinguish alternatives.
* Reintroducing sign by hand produces instability near zero.
* The latent structure being normalized is unclear.

These issues are not numerical artifacts. They indicate a mismatch between the role of softmin and the role traditionally played by activation functions.

---

## Where EM Actually Lives

Expectation–maximization does not act on individual coordinates in isolation. It acts on **collections of mutually exclusive or competing hypotheses**.

Softmin is therefore not an activation in the usual sense. It is a **competition operator**. It belongs wherever alternatives must be compared and normalized against one another:

* mixture components,
* attention keys,
* classes,
* experts,
* or symmetric alternatives such as positive versus negative sides of a hyperplane.

Replacing activations with softmin implicitly assumes that each scalar output represents a hypothesis set. In general, it does not.

---

## Activations as EM Summaries

Once this distinction is made, the roles separate cleanly:

* Softmin induces EM by normalizing across hypotheses and producing responsibilities.
* Activations are **summaries of the resulting posterior state**, passed forward to subsequent layers.

Preserving sign illustrates this clearly. Sign information corresponds to two competing hypotheses: positive and negative. When these alternatives are made explicit and normalized via softmin, the activation is no longer arbitrary. It is a posterior statistic of that competition.

This also explains why operations such as absolute value and ReLU are incompatible with EM. They enforce hard assignment: one alternative receives zero responsibility and zero gradient. EM, by contrast, requires soft competition everywhere.

---

## The Role of Tanh

For a symmetric signed kernel—such as a Gaussian with means at ( \pm m )—normalization across the two hypotheses yields responsibilities (r_+) and (r_-). The quantity naturally propagated forward is not the full responsibility vector, but its posterior mean:
[
\mathbb{E}[s \mid z] = r_+ - r_-.
]

This posterior mean has the functional form of tanh (up to scale). In this sense, tanh is not a heuristic nonlinearity, but the appropriate summary of an implicit EM step over signed alternatives.

---

## Reframing the Design Question

The original intuition—that activations should be replaced by softmin to induce EM—was close, but imprecise. The more accurate statement is:

> **Softmin belongs between hypotheses.
> Activations are posterior summaries of that competition.**

EM structure is enforced by the objective and its normalization. Activations reflect the geometry of the kernel and the latent hypothesis space, not arbitrary design choices.

This reframing shifts attention away from inventing new nonlinearities and toward understanding which kernels and hypothesis structures are being assumed. Once those are fixed, the form of the activation is determined.

The purpose of this paper is to make that structure explicit.
