# Implicit EM: Assumptions and Scope

This note makes explicit the assumptions under which **implicit expectation–maximization (EM)** arises in neural models, and clarifies the boundaries of applicability of the framework developed in this project.

The goal is not to broaden the claim, but to state precisely **when EM is forced by the objective geometry and when it is not**.

---

## What Is Meant by “Implicit EM”

By *implicit EM* we mean the following:

* No explicit latent variables are introduced.
* No alternating E-step / M-step algorithm is run.
* No probabilistic model is instantiated at the architectural level.

Instead, EM structure arises as a **mathematical consequence** of:

* exponentiating energies,
* normalizing across alternatives,
* and optimizing the resulting objective via gradients.

Responsibilities are not auxiliary variables. They are normalized exponentials and, equivalently, gradients of a log-sum-exp objective with respect to energies.

---

## Core Structural Assumptions

Implicit EM arises if and only if the following structural conditions are satisfied.

### 1. Discrete Hypothesis Set

There must exist a discrete index (h) representing **competing hypotheses**.

Examples include:

* mixture components,
* classes,
* attention keys,
* experts,
* symmetric alternatives (e.g. sign).

Without discrete alternatives, there is nothing to normalize over and no notion of responsibility.

---

### 2. Energy-Based Representation

Each hypothesis (h) must be associated with a scalar **energy**:
[
d_h(z)
]
interpreted as a distance or negative log-likelihood.

Lower energy corresponds to higher plausibility.

The energy need not be interpretable probabilistically, but it must be well-defined and comparable across hypotheses.

---

### 3. Exponentiation of Energies

Energies must be converted into unnormalized likelihoods via exponentiation:
[
P_h(z) = \exp(-d_h(z)).
]

This step is essential. Linear or polynomial mappings do not induce EM structure. Exponentiation is what enables competition to be expressed multiplicatively.

---

### 4. Normalization Across Hypotheses

Unnormalized likelihoods must be normalized across hypotheses:
[
r_h(z) = \frac{\exp(-d_h(z))}{\sum_{h'} \exp(-d_{h'}(z))}.
]

This normalization is the implicit E-step. It produces responsibilities that sum to one and encode soft assignment.

Any construction lacking normalization is not EM, regardless of surface similarity.

---

### 5. Log-Sum-Exp Objective

The canonical objective inducing this normalization is:
[
L(z) = \log \sum_h \exp(-d_h(z)).
]

Optimizing this objective guarantees:

* normalized responsibilities,
* smooth competition,
* responsibility-weighted gradients.

Other objectives may approximate EM behavior, but log-sum-exp is the minimal and exact form.

---

## Consequences of These Assumptions

When the above conditions are met:

* Responsibilities emerge deterministically from the forward pass.
* Gradients with respect to energies are exactly the negative responsibilities.
* Learning proceeds as a continuous analogue of EM.
* No explicit inference algorithm is required.

These properties are invariant to the specific form of the energy kernel.

---

## What Breaks Implicit EM

The following violate one or more of the required conditions.

### Independent Nonlinearities

Applying elementwise nonlinearities (e.g. sigmoid, tanh, ReLU) **without normalization across alternatives** does not induce EM. Such operations may be probabilistically interpretable, but they do not produce responsibilities.

---

### Hard Gating

Operations that enforce exact zeros (e.g. ReLU, max, hard attention) implement hard assignment.

From the EM perspective, this corresponds to component collapse: one hypothesis receives full responsibility, others receive none and cannot recover.

---

### Unnormalized Energies

Using exponentiated energies without normalization produces scores, not posteriors. EM requires relative, not absolute, likelihoods.

---

### Continuous Hypothesis Spaces

Implicit EM, as formulated here, applies to discrete hypothesis sets. Continuous latent variables require different machinery and are not covered by this framework.

---

## Closed-World Assumption

Normalization enforces a closed-world assumption:
[
\sum_h r_h = 1.
]

All probability mass is assigned to known hypotheses. Rejection, abstention, or “none of the above” behavior requires additional structure and breaks standard EM.

---

## Scope of the Present Work

This project assumes:

* discrete hypothesis sets,
* energy-based kernels,
* exponentiation and normalization,
* gradient-based optimization.

Within this scope, EM structure is unavoidable and activations arise as posterior summaries.

Outside this scope, EM may not apply, and no such claim is made.

---

## Summary

Implicit EM is not a metaphor.
It is a structural consequence of specific design choices.

Those choices are explicit, minimal, and restrictive.
This note exists to state them clearly.
