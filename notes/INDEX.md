# Index

This repository develops a framework in which neural activations are understood as **posterior statistics** arising from **implicit expectation–maximization (EM)** induced by distance-based objectives with log-sum-exp structure. The notes progress from motivation and discovery, through formal assumptions and derivations, to boundary cases and architectural implications.

---

## High-Level Overview and Motivation

**`overview.md`**
High-level overview of the core claim: neural activations are posterior summaries arising from implicit EM induced by log-sum-exp objectives, not arbitrary nonlinearities. Establishes the unifying perspective, assumptions, and scope of the project.

**`motivation.md`**
Explains why standard activations like ReLU are incompatible with distance-based reasoning and why naive replacements like softmin fail. Clarifies where EM structure actually lives and reframes activations as posterior summaries rather than competition operators.

**`discovery.md`**
Narrative account of how failed attempts to engineer a “softmin ReLU” led to re-deriving tanh and uncovering the implicit EM framework. Provides intuition and historical context rather than formal theory.

---

## Structural Foundations of Implicit EM

**`implicit_em_assumptions.md`**
Precisely states the minimal structural assumptions under which implicit EM arises (discrete hypotheses, energies, exponentiation, normalization). Clearly delineates what does and does not count as EM in neural models.

**`competition_vs_summary.md`**
Clarifies the central distinction between competition (softmin / normalization across hypotheses) and summary (activation). Explains why confusing these roles leads to incorrect architectures and broken EM interpretations.

---

## Kernels, Energies, and Geometry

**`kernels_for_implicit_em.md`**
Catalog of energy and distance kernels compatible with implicit EM, including Gaussian, Laplace, Student-t, and learned energies. Explains how kernel choice determines robustness, geometry, and activation behavior.

**`generalized_derivation.md`**
A step-by-step template for deriving an activation function from a kernel via implicit EM. Formalizes the pipeline: energies → responsibilities → posterior statistic → activation.

---

## Activations as Posterior Statistics

**`posterior_statistics.md`**
Explains activations as choices of posterior statistics (mean, full responsibility vector, mode, etc.). Shows how tanh, softmax, attention, and ReLU correspond to different ways of summarizing the same posterior.

**`act_tanh_from_signed_gaussian.md`**
Concrete derivation of tanh as the posterior mean of a signed Gaussian (Mahalanobis) distance model. Serves as the canonical worked example of the general derivation pattern.

---

## Scale, Degeneracy, and Failure Modes

**`temperature_and_scale.md`**
Explains the role of temperature (or scale) in implicit EM and how it controls sharpness, robustness, saturation, and hard-assignment limits. Interprets temperature as a semantic inference parameter rather than a tuning trick.

**`why_relu_breaks_em.md`**
Shows why ReLU is inference-breaking under the implicit EM framework, due to hard assignment and zero responsibility. Clarifies where EM structure exists in typical ReLU networks and where it does not.

---

## Architectural Boundaries and Extensions

**`open_world_and_rejection.md`**
Analyzes the closed-world assumption enforced by normalization in implicit EM and explains why softmax-style models cannot express true rejection or “none of the above.” Defines the limits of the framework for open-world tasks.

**`relationship_to_attention.md`**
Situates attention mechanisms within the same competition–summary structure as implicit EM. Shows that attention is an explicit posterior inference followed by a posterior mean, differing from scalar activations mainly in dimensionality and information preservation.

---

## Meta / Positioning

**`titles.md`**
Meta discussion of paper title options and framing strategies, analyzing how different titles signal scope, claims, and reviewer expectations. Useful for positioning the work, not for technical content.

---


## Extending Implicit EM to Internal Layers

**`where_em_lives.md`**
Clarifies that the implicit EM identity ∂L/∂dⱼ = −rⱼ holds at the model-output interface. Internal layers receive responsibility-weighted gradients but are not themselves performing mixture inference. Distinguishes "EM-structured loss" from "EM throughout the network."

**`competition_within_layers.md`**
Identifies the core problem: ReLU layers have no competition among units. Gradients flow independently. Multiple hyperplanes can learn identical cuts with nothing pushing them apart. Contrasts with output-layer softmax where units must compete for responsibility.

**`layer_wise_implicit_em.md`**
Explores whether implicit EM can be induced at each layer, not just at the output. Two approaches: (1) greedy layer-wise pretraining with LSE objectives, (2) activation structures whose backward pass has responsibility-weighted gradients. Analyzes tradeoffs between local specialization and global task alignment.

**`candidate_competitive_activations.md`**
Catalogs activation functions that might introduce within-layer competition: softmax, log-softmax, z ⊙ softmax(z), grouped softmax. Analyzes gradient structure of each. Evaluates expressiveness vs. specialization tradeoff.

**`volume_and_collapse_in_activations.md`**
The collapse problem: without volume penalties, competitive activations allow single units to dominate. In GMMs, log-determinant prevents this. For activations, candidates include weight normalization, learned temperature, entropy regularization, or explicit log-det penalties.

**`infomax_as_em_objective.md`**
The synthesis: InfoMax is the objective for implicit EM. Softmax provides the mechanism (responsibility-weighted updates). InfoMax provides the goal (independent, informative features). The marginal entropy term prevents dead units; the total correlation term prevents redundancy and collapse. InfoMax is the principled volume control.

**`experimental_design_single_layer.md`**
Concrete experimental setup: 2D input, single-layer autoencoder, K hidden units. Compare ReLU, softmax, z ⊙ softmax(z), log-softmax. Metrics: hyperplane redundancy, reconstruction error, region specialization.

