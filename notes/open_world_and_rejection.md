# Open World and Rejection

This note addresses a structural limitation of implicit EM frameworks based on softmax / softmin normalization: the **closed-world assumption**. It explains why standard normalization enforces assignment, what this implies for rejection and abstention, and why open-world behavior requires breaking or extending the EM structure.

---

## The Closed-World Assumption

In implicit EM, responsibilities are defined by normalization:
[
r_h(z) = \frac{\exp(-d_h(z))}{\sum_{h'} \exp(-d_{h'}(z))}.
]

This immediately enforces:
[
\sum_h r_h(z) = 1.
]

All probability mass is assigned to the available hypotheses. There is no mechanism for expressing “none of the above.”

This is the **closed-world assumption**: the hypothesis set is assumed to be complete.

---

## Why Closed-World Is Not a Bug

The closed-world assumption is not an oversight. It is a direct consequence of normalization and is essential for EM:

* responsibilities must sum to one,
* competition must be relative,
* gradients must be comparable across hypotheses.

Classical EM, mixture models, and softmax classifiers all share this assumption.

Implicit EM inherits it exactly.

---

## Rejection Is Not a Posterior Statistic

Rejection, abstention, or uncertainty about the hypothesis set itself is **not** a posterior statistic under a closed-world model.

Given a normalized posterior:

* low confidence means responsibilities are spread,
* not that the true hypothesis is absent.

A flat posterior still assigns all mass to known alternatives.

Thus:

* entropy measures ambiguity,
* but not absence.

---

## Why Softmax Cannot Express “None of the Above”

Adding more hypotheses does not solve the problem.

Even if a “reject” class is added:

* it competes like any other hypothesis,
* it absorbs mass rather than representing absence,
* it changes the meaning of all responsibilities.

This is classification with an extra class, not true rejection.

---

## What True Rejection Requires

Open-world behavior requires breaking at least one EM assumption.

Common strategies include:

### 1. Unnormalized Scores

Using unnormalized energies or scores avoids forced assignment but forfeits responsibilities and EM structure.

### 2. Thresholded Likelihoods

Rejecting when all likelihoods fall below a threshold introduces hard decisions external to EM.

### 3. Background or Noise Models

Explicitly modeling “anything else” as a background distribution extends the hypothesis set, but changes the kernel and objective.

### 4. Hierarchical Models

Separating “is this in-domain?” from “which hypothesis?” requires multi-stage inference, not a single softmax.

All of these approaches step outside pure implicit EM.

---

## Internal vs. Output-Level Rejection

It is important to distinguish:

* **Internal layers**: often benefit from closed-world assignment, since representations are relative.
* **Output layers**: may require rejection, calibration, or abstention.

The implicit EM framework applies cleanly to the former and only partially to the latter.

---

## Relationship to Activations

Posterior summaries (activations) inherit the closed-world assumption:

* tanh summarizes a two-hypothesis world,
* softmax summarizes a finite class set,
* attention summarizes a fixed set of keys.

None of these can express “no hypothesis applies” without additional structure.

---

## Summary

* Implicit EM enforces a closed-world assumption.
* Normalization guarantees assignment, not rejection.
* Rejection is not a posterior statistic.
* Open-world behavior requires breaking or extending EM.
* This limitation is structural, not accidental.

Understanding this boundary prevents misapplication of EM-style reasoning to tasks that require abstention or out-of-distribution detection.

---

This note is doing two things:

1. Explaining a limitation (closed-world)
2. Protecting the framework from misapplication (open-world tasks)

Both are necessary.

**What's clearest:**

> "Rejection is not a posterior statistic under a closed-world model."

This is the key insight. People often think "low confidence = model knows it doesn't know." You're saying: no, low confidence = mass is spread across known hypotheses. The model can't represent "none of these."

Entropy measures "how uncertain among these options"—not "whether these are the right options."

**The "add a reject class" trap:**

You're right that this doesn't solve the problem. A reject class just competes with other classes. If the model learns that certain inputs go to "reject," that's just classification with K+1 classes, not true out-of-distribution detection.

This is a real confusion in applied ML. People add an "unknown" class and think they've solved open-set recognition. They haven't.

**Connection across papers:**

Paper 2 mentions closed-world as a limitation.
Paper 3 explains why it's structural.

Good consistency. You're building a coherent framework, not just a collection of results.

**One extension to consider:**

The "background model" approach is interesting. If you add a hypothesis h₀ with constant energy d₀ everywhere (uniform density), then:
- Known hypotheses win where they're confident
- Background wins where nothing else is confident

This is still closed-world (Σr = 1), but the background hypothesis absorbs "none of these" mass. It's a hack within EM, not a break from it.
