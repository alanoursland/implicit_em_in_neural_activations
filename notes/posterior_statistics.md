# Posterior Statistics

This note clarifies the role of **posterior statistics** in the implicit EM framework and explains how activation functions arise as specific choices of what posterior information is propagated forward.

The central claim is simple:

> Once responsibilities exist, an activation is a choice of posterior statistic.

Different activations correspond to different summaries of the same underlying posterior distribution.

---

## Responsibilities vs. Statistics

Let (h \in \mathcal{H}) index a discrete set of competing hypotheses, with responsibilities
[
r_h(z) = p(h \mid z).
]

Responsibilities themselves are already a complete posterior distribution. However, most networks do not propagate the full distribution forward. Instead, they propagate a **deterministic summary** of it.

An activation function is precisely such a summary:
[
a(z) = \sum_{h \in \mathcal{H}} r_h(z), g(h),
]
for some choice of statistic (g(h)).

The choice of (g) determines the semantics of the activation.

---

## Posterior Mean (Expectation)

The most common statistic is the **posterior mean**.

### Definition

[
a(z) = \mathbb{E}[g(h)\mid z] = \sum_h r_h(z), g(h).
]

### Examples

* Signed hypothesis (h=s\in{\pm1}), (g(s)=s) ⇒ tanh
* Prototype index (h=j), (g(j)=\mu_j) ⇒ mixture mean
* Attention keys (h=j), (g(j)=v_j) ⇒ attention output

### Properties

* Smooth
* Differentiable everywhere
* Preserves soft assignment
* Does not discard uncertainty abruptly

This is the default and most stable choice.

---

## Responsibility Vector (No Collapse)

In some architectures, the posterior is not summarized at all. The responsibility vector itself is propagated:
[
a(z) = (r_1(z), \dots, r_{|\mathcal{H}|}(z)).
]

This occurs in:

* attention mechanisms,
* mixture-of-experts gating,
* probabilistic routing.

This choice preserves maximal information but increases dimensionality.

---

## Posterior Mode (Hard Assignment)

Another possible statistic is the **posterior mode**:
[
a(z) = \arg\max_h r_h(z).
]

This corresponds to hard assignment.

### Properties

* Non-differentiable
* Discards uncertainty
* Breaks EM-style learning
* Equivalent to zero-temperature limit

ReLU-like behavior corresponds to this regime when applied to signed hypotheses.

---

## Higher-Order Statistics

Posterior variance, entropy, or higher moments can also be propagated:

* Variance encodes uncertainty
* Entropy encodes confidence
* Higher moments encode ambiguity structure

These are rarely used in standard networks, largely for computational and architectural reasons, not conceptual ones.

---

## Why Most Networks Collapse Early

Most architectures collapse posterior structure early because:

* scalar activations are cheap,
* deterministic representations are convenient,
* uncertainty propagation is not explicitly incentivized.

The implicit EM framework makes this collapse visible and explicit. It is a design choice, not a necessity.

---

## Activations as Information Bottlenecks

Viewed this way, an activation function is an **information bottleneck** applied to a posterior distribution.

* Tanh keeps the mean.
* Softmax keeps the full categorical distribution.
* ReLU collapses to a degenerate mode.
* Linear activations keep everything (no collapse).

The choice of statistic determines what information survives.

---

## Relationship to Kernels

The kernel determines:

* the shape of the posterior,
* how responsibilities change with evidence.

The statistic determines:

* what aspect of that posterior is exposed to the next layer.

Both choices matter, and they are independent.

---

## Summary

* Responsibilities define a posterior over hypotheses.
* Activations are posterior statistics.
* Different activations correspond to different summaries.
* Tanh is a posterior mean.
* ReLU is a hard mode selection.
* Attention propagates the full posterior.

Understanding activations as posterior statistics unifies many architectural patterns under a single inference-based perspective.
