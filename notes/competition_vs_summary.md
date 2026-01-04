# Competition vs. Summary

This note clarifies a central conceptual distinction that underlies the entire project: the difference between **competition** and **summary**. Conflating these two roles leads directly to incorrect architectures and misinterpretations of EM structure in neural networks.

---

## The Tempting but Incorrect Idea

A natural response to the implicit EM perspective is the following proposal:

> Replace activation functions with softmin to make EM explicit at the level of each linear layer.

The intuition is understandable. Softmin converts distances into normalized likelihoods, and normalized likelihoods resemble responsibilities. If EM arises from log-sum-exp normalization, then applying softmin everywhere appears to expose EM locally.

This idea is wrong.

Not because softmin is inappropriate, but because it is being asked to play the wrong role.

---

## What Softmin Actually Does

Softmin is a **competition operator**.

Given a *set* of energies ({d_h}) indexed by hypotheses (h), softmin performs:

[
r_h = \frac{\exp(-d_h)}{\sum_{h'} \exp(-d_{h'})}
]

This operation has a precise meaning:

* it assumes the existence of multiple, mutually exclusive alternatives,
* it normalizes evidence across those alternatives,
* it produces responsibilities.

Softmin does **not** transform a single scalar into something meaningful. Without alternatives, there is nothing to normalize, and no notion of responsibility can arise.

---

## What Activations Do

Activation functions serve a different purpose.

They do not induce competition.
They **summarize the outcome of competition**.

An activation is a deterministic function of the posterior state produced by normalization. It collapses responsibility-weighted structure into a representation that can be propagated forward.

This distinction is essential:

* **Softmin**: compares alternatives, induces EM.
* **Activation**: summarizes the posterior, propagates information.

These are not interchangeable operations.

---

## Where EM Lives

Expectation–maximization operates *between* representation and activation, not *inside* the activation.

The correct conceptual pipeline is:

```
Linear layer → z (distance-like coordinate)
                ↓
        Hypotheses exist (e.g. s ∈ {+1, −1})
                ↓
        Kernel assigns energy to each hypothesis
                ↓
        Softmin / log-sum-exp normalization
                ↓
        Responsibilities (implicit E-step)
                ↓
        Posterior statistic
                ↓
        Activation
```

EM lives in the middle of this chain. The activation is what comes out the other side.

Applying softmin directly to the output of a linear unit skips the hypothesis layer entirely. The resulting pathologies—loss of sign, instability near zero, unclear semantics—are consequences of this omission, not implementation errors.

---

## Why Sign Matters

A single signed scalar already implies a latent hypothesis structure: positive versus negative. Preserving this structure requires treating the two sides as distinct hypotheses and normalizing across them.

If this competition is ignored or collapsed prematurely:

* sign must be reintroduced manually,
* gradients become unstable,
* EM structure is destroyed.

When the hypothesis structure is made explicit, softmin produces responsibilities over alternatives, and the activation becomes a posterior summary of that distribution.

---

## Tanh and ReLU Revisited

This distinction immediately clarifies the behavior of familiar nonlinearities.

* **Tanh** corresponds to soft competition between two symmetric hypotheses, followed by propagation of the posterior mean. Both hypotheses retain nonzero responsibility everywhere. EM dynamics are preserved.

* **ReLU** corresponds to hard assignment. One hypothesis receives full responsibility, the other receives none. Once inactive, a hypothesis receives no gradient signal. From the EM perspective, this is a degenerate limit in which one component dies.

ReLU is therefore not merely sparse; it is inference-breaking.

---

## What This Project Is Not Doing

This work does **not** propose replacing activation functions with softmin.

It explains why that proposal is ill-posed.

Softmin belongs where hypotheses compete. Activations belong where posterior information is summarized. Confusing these roles leads to architectures that appear EM-like but are not.

By making this distinction explicit, the remainder of the project focuses on the correct question:

> Given a kernel and a hypothesis structure, what posterior statistic does the network propagate forward?

That question has a principled answer. This note exists to prevent asking the wrong one.
