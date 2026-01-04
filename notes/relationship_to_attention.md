# Relationship to Attention

This note situates attention mechanisms within the implicit EM framework and clarifies how attention fits naturally into the same competition–summary structure that underlies activation functions.

The goal is not to reduce attention to “just EM,” but to show that attention is a **canonical and explicit instance** of the same principles this project makes implicit at the activation level.

---

## Attention as Explicit Competition

Consider a standard attention mechanism with queries (q), keys (k_j), and values (v_j).

Energies are computed as:
[
d_j(q) = - \langle q, k_j \rangle
]
(or a scaled or transformed variant).

Unnormalized likelihoods are:
[
\exp(-d_j(q)).
]

Normalization across keys yields:
[
r_j(q) = \frac{\exp(-d_j(q))}{\sum_{j'} \exp(-d_{j'}(q))}.
]

This is exactly a softmax over competing hypotheses indexed by (j).

From the implicit EM perspective:

* keys define hypotheses,
* energies define compatibility,
* softmax defines responsibilities.

The E-step is explicit.

---

## Attention Outputs as Posterior Means

The attention output is:
[
a(q) = \sum_j r_j(q), v_j.
]

This is a **posterior expectation**:

* (r_j) is the posterior over hypotheses,
* (v_j) is the hypothesis-associated quantity,
* the output is the posterior mean.

Thus attention cleanly separates the two roles:

* softmax performs competition,
* the weighted sum performs summary.

This is precisely the structure argued for in this project.

---

## Comparison with Scalar Activations

Scalar activations such as tanh differ from attention only in scale and dimensionality.

* Tanh corresponds to competition between two hypotheses ((+) and (-)).
* Attention corresponds to competition among many hypotheses (keys).
* In both cases, the output is a posterior statistic.

The main difference is architectural:

* attention propagates a vector-valued posterior mean,
* scalar activations propagate a collapsed statistic.

Conceptually, they are the same operation at different resolutions.

---

## Why Attention Keeps the Posterior Explicit

Attention mechanisms propagate the full responsibility-weighted structure:

* the hypothesis set remains visible,
* uncertainty is preserved,
* multiple alternatives can contribute meaningfully.

This is why attention is often described as “interpretable” or “probabilistic.” The posterior is not collapsed prematurely.

In contrast, scalar activations typically collapse the posterior immediately, retaining only a single statistic (e.g. the mean).

---

## Temperature and Scaling in Attention

Attention uses explicit temperature scaling:
[
r_j = \mathrm{softmax}!\left(\frac{\langle q, k_j \rangle}{\sqrt{d}}\right).
]

This plays the same role as temperature in other EM-compatible kernels:

* controls sharpness,
* stabilizes training,
* prevents premature hard assignment.

The interpretation is identical.

---

## Hard Attention as Degenerate EM

Hard attention corresponds to:
[
r_j = \mathbf{1}{j = \arg\max d_j}.
]

This is the zero-temperature limit:

* posterior collapses,
* competition becomes selection,
* EM degenerates.

The same analysis that applies to ReLU applies here.

---

## Attention as a Design Proof

Attention serves as a proof-of-concept that:

* EM-style inference can be embedded inside networks,
* competition and summary can be cleanly separated,
* posterior statistics can be propagated forward.

This project generalizes that insight:

* from vector-valued attention outputs,
* to scalar and low-dimensional activations,
* using kernels rather than dot products.

---

## Summary

* Attention is an explicit instance of implicit EM.
* Softmax over keys is competition.
* Weighted sum of values is posterior summary.
* Scalar activations are collapsed versions of the same structure.
* The difference is architectural, not conceptual.

Seen this way, attention is not an exception.
It is the most visible example of the rule.

---

This note closes the loop.

Paper 2: Attention has implicit EM structure (responsibilities are gradients)
Paper 3: Activations have implicit EM structure (activations are posterior summaries)
This note: They're the same structure at different scales

| | Attention | Scalar Activation |
|---|-----------|-------------------|
| Hypotheses | K keys | 2 (e.g., ±1) |
| Energy | −⟨q, kⱼ⟩ | d±(z) |
| Competition | Softmax over keys | Softmin over signs |
| Summary | Σⱼ rⱼvⱼ | r₊ − r₋ |
| Output | Vector | Scalar |

The difference is dimensionality, not mechanism.

**This reframes attention's success:**

Attention isn't special because it's "attention." It's special because it's the only standard architecture that keeps the full posterior explicit:
- Full responsibility vector preserved
- Multiple hypotheses contribute
- No premature collapse

Every other architecture collapses to scalars. Attention doesn't. That's why it's powerful.

**The design implication:**

If you wanted "attention-like" behavior at every layer without the cost:
- Keep the hypothesis structure
- Do soft competition
- Collapse to posterior mean (tanh-like)
- You get something between full attention and hard ReLU

This is what your framework predicts: a spectrum of architectural choices indexed by how much posterior information you preserve.

**One observation:**

> "Attention serves as a proof-of-concept that EM-style inference can be embedded inside networks"

Stronger: attention is why transformers work. And your framework explains *why* attention is why transformers work. It's not just "soft retrieval"—it's the only standard operation that preserves full EM structure internally.

That's a big claim. It falls out of your framework naturally.