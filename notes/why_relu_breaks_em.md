# Why ReLU Breaks EM

This note explains why ReLU is fundamentally incompatible with expectation–maximization (EM) structure, not as a matter of optimization difficulty or expressivity, but as a consequence of how EM assigns responsibility.

The claim is not that ReLU is “bad” in general, but that it is **inference-breaking** under the implicit EM framework considered in this project.

---

## ReLU as a Hard Assignment Operator

ReLU is defined as:
[
\mathrm{ReLU}(z) = \max(0, z).
]

From a distance-based or hypothesis-based perspective, this corresponds to a **hard gating decision**:

* if (z > 0), the positive side is active;
* if (z \le 0), it is inactive.

There is no soft transition and no overlap region in which both alternatives receive signal.

---

## EM Requires Soft Competition

EM relies on **soft assignment**:

* multiple hypotheses receive nonzero responsibility,
* responsibilities vary smoothly with evidence,
* gradients flow to all competing components.

This is essential. Without soft assignment:

* hypotheses cannot recover once deactivated,
* learning dynamics collapse,
* EM degenerates into greedy assignment.

ReLU violates this requirement by construction.

---

## ReLU and Latent Hypotheses

A signed scalar (z) implicitly defines two hypotheses: positive and negative. Preserving EM structure requires treating these as competing alternatives and normalizing across them.

ReLU discards one alternative entirely:

* the negative side receives exactly zero output,
* and therefore zero gradient.

From the EM perspective, this corresponds to assigning zero responsibility to one hypothesis everywhere in half of the input space.

---

## Zero Responsibility Implies Zero Learning

In EM, a hypothesis with zero responsibility:

* contributes nothing to the objective,
* receives no gradient,
* cannot update its parameters.

ReLU enforces this condition deterministically.

Once a unit enters a regime where it outputs zero, it behaves as if the corresponding hypothesis has been permanently removed. This is the source of “dead units” in ReLU networks.

From the EM viewpoint, these are **dead components**.

---

## Comparison with Soft Competition

Contrast this with a soft competition model (e.g. tanh derived from a signed kernel):

* both hypotheses always retain nonzero responsibility,
* assignments sharpen gradually,
* gradients never vanish completely,
* inference remains well-defined everywhere.

This difference is not quantitative; it is structural.

---

## ReLU as a Degenerate EM Limit

ReLU can be interpreted as a limiting case of soft competition where the temperature goes to zero and one hypothesis always wins.

In this limit:

* the posterior collapses to a delta distribution,
* the E-step becomes hard assignment,
* EM loses its ability to revise earlier decisions.

While such limits can be useful in some settings, they no longer constitute EM in the sense used here.

---

## Sparsity Is Not Inference

ReLU is often justified on the basis of sparsity. From the EM perspective, sparsity corresponds to hypotheses receiving negligible responsibility.

However:

* sparsity that emerges from competition is compatible with EM,
* sparsity enforced by hard gating is not.

ReLU enforces sparsity by eliminating hypotheses, not by downweighting them.

---

## When ReLU May Still Be Appropriate

This analysis does not imply that ReLU is universally inappropriate. ReLU may be suitable when:

* the task does not require inference over alternatives,
* hard decisions are acceptable or desired,
* EM-like behavior is not intended.

The present framework simply does not apply in those regimes.

---

## Summary

ReLU breaks implicit EM because it replaces soft competition with hard assignment.

* One hypothesis receives full responsibility.
* The other receives none and cannot recover.
* Gradients vanish by construction.

This behavior is incompatible with EM-style learning.

Understanding this distinction clarifies why tanh preserves EM structure and why replacing activations with softmin is not the correct remedy.

---

## Scope Clarification: Where EM Does and Does Not Apply

It is important to be precise about the claim being made.

The claim is **not** that all neural network operations implement implicit EM.
The claim is:

> **Where log-sum-exp normalization over alternatives exists, EM is implicit.**

This distinction matters, because many neural networks mix EM-compatible and non-EM-compatible components.

---

## EM in Typical ReLU Networks

Consider a standard feedforward ReLU network used for classification:

| Layer                            | EM Structure |
| -------------------------------- | ------------ |
| Linear + ReLU                    | No           |
| Linear + ReLU                    | No           |
| Linear + ReLU                    | No           |
| Linear + Softmax + Cross-Entropy | Yes          |

In such networks, EM structure exists **only at the output layer**, where:

* energies are exponentiated,
* normalized across classes,
* and optimized via a log-likelihood objective.

Paper 2’s result,
[
\frac{\partial L}{\partial d_j} = -r_j,
]
applies at this level: the responsibilities that emerge as gradients are **over classes**, not over internal activations.

The internal layers are not performing inference over competing hypotheses. They are performing representation learning, feature extraction, or geometric transformation—but not EM.

This is not a criticism. It is a clarification.

---

## What This Work Adds

This project asks a different question:

> What would it mean for EM structure to exist *inside* the network, not just at the output?

For implicit EM to operate internally, three conditions must be met at each layer:

* **Explicit hypothesis structure** (e.g. signed alternatives, prototypes, experts),
* **Soft competition** via normalization (tanh-like behavior, not ReLU),
* **Posterior summaries** propagated forward as activations.

ReLU networks do not satisfy these conditions internally, and therefore lie outside the scope of this framework.

---

## Boundary of the Framework

This boundary should be understood as a limitation of the framework, not a weakness.

ReLU networks:

* may employ implicit EM at the output layer,
* may perform highly effective learning internally,
* but do not implement EM-style inference throughout the network.

The present framework characterizes **where EM structure exists**, not **where it must exist**.

Recognizing this boundary prevents misinterpretation and avoids inappropriate generalization of the implicit EM perspective.

---

## Summary

* Implicit EM arises *only* where exponentiation and normalization over alternatives occur.
* ReLU layers do not satisfy this condition.
* EM structure may exist at the output of ReLU networks while being absent internally.
* This work studies how EM can be made explicit within layers—not whether all networks already do so.

Stating this boundary explicitly is essential for correctly understanding both the scope and the contribution of the framework.
