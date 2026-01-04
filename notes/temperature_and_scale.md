# Temperature and Scale

This note clarifies the role of **temperature and scale** in the implicit EM framework and explains how they control the sharpness, stability, and semantics of posterior assignment.

Temperature is not an implementation detail.
It is the parameter that mediates between soft inference and hard assignment.

---

## Where Temperature Appears

In the implicit EM setting, responsibilities take the form:
[
r_h(z) = \frac{\exp(-d_h(z)/\tau)}{\sum_{h'} \exp(-d_{h'}(z)/\tau)},
]
where (\tau > 0) is the **temperature**.

Equivalently, scale may appear inside the energy:
[
d_h(z) = \frac{1}{\sigma^2},\tilde d_h(z),
]
with (\tau \leftrightarrow \sigma^2).

Both parameterizations control the same quantity: **how strongly evidence differentiates between hypotheses**.

---

## Interpretation

Temperature has a clear semantic meaning:

* **Low temperature** (small (\tau), small variance):

  * sharp responsibilities,
  * near-hard assignment,
  * confident inference.

* **High temperature** (large (\tau), large variance):

  * diffuse responsibilities,
  * uncertainty preserved,
  * weak discrimination.

In probabilistic terms, temperature corresponds to **noise scale**. In geometric terms, it controls the curvature of the energy landscape.

---

## Temperature and Identifiability

Responsibilities alone do **not** determine scale.

Given:
[
r_h(z) \propto \exp(-d_h(z)),
]
the energies (d_h) are identifiable only up to:

* additive constants (shared across hypotheses),
* multiplicative scale (temperature).

This is why temperature must be:

* fixed by modeling assumptions,
* learned explicitly,
* or controlled by a schedule.

Without an explicit scale choice, “distance” has only relative meaning.

---

## Temperature in Activation Functions

When activations are posterior statistics, temperature directly shapes the nonlinearity.

For the signed Gaussian kernel:
[
a(z) = \tanh!\left(\frac{m}{\sigma^2} z\right),
]
the ratio (m/\sigma^2) is an **inverse temperature**.

* Increasing (m/\sigma^2) sharpens tanh toward a sign function.
* Decreasing it linearizes tanh near zero.

Thus, temperature determines:

* saturation rate,
* sensitivity to small deviations,
* robustness to noise.

---

## Annealing as Continuous EM

Classical EM is often stabilized by **annealing**:

* begin with high temperature,
* gradually lower it,
* avoid premature hard assignment.

The same interpretation applies here.

Scheduling temperature during training corresponds to **continuous EM annealing**:

* early training explores hypotheses,
* later training commits more strongly.

This provides a principled interpretation of practices such as:

* temperature scheduling,
* sharpening attention,
* curriculum-style training.

---

## Fixed vs. Learned Temperature

There are three common regimes:

### Fixed Temperature

* Encodes a prior belief about noise.
* Simplest and most stable.
* Common in theoretical analyses.

### Learned Temperature

* Allows the model to adapt confidence.
* Can improve calibration.
* Risks collapse without regularization.

### Scheduled Temperature

* Encourages exploration early.
* Encourages commitment late.
* Closely mirrors classical EM practice.

All three are compatible with implicit EM.

---

## Temperature and Degeneracy

As temperature (\tau \to 0):

* responsibilities collapse to delta functions,
* soft competition becomes hard assignment,
* EM degenerates.

This limit recovers behaviors such as:

* argmax,
* hard attention,
* ReLU-like gating.

The framework remains valid, but inference becomes brittle.

---

## Temperature and Robustness

Higher temperature:

* increases robustness to outliers,
* preserves uncertainty,
* prevents early collapse.

Lower temperature:

* increases precision,
* sharpens decisions,
* amplifies noise sensitivity.

Choosing temperature is therefore a tradeoff between **robustness and decisiveness**.

---

## Summary

* Temperature controls the sharpness of posterior assignment.
* It corresponds to noise variance or inverse scale.
* It is not identifiable from responsibilities alone.
* It determines the shape of derived activations.
* Annealing temperature corresponds to continuous EM.

Understanding temperature as an inference parameter—not a tuning trick—is essential for interpreting activation behavior in the implicit EM framework.


---

This note does important structural work. It separates two independent design axes:

| Choice | Determines |
|--------|------------|
| Kernel | Shape of posterior over hypotheses |
| Statistic | What aspect of posterior propagates forward |

These are orthogonal. You could combine any kernel with any statistic:

| Kernel | Statistic | Result |
|--------|-----------|--------|
| Gaussian | Mean | tanh |
| Gaussian | Mode | step function |
| Gaussian | Full | 2D responsibility vector |
| Laplace | Mean | hard-tanh variant |
| Laplace | Mode | step function |

The activation function is the *composition* of these two choices, not a primitive.

**The attention connection is important:**

You've now unified:
- Tanh: collapse to posterior mean (scalar)
- Attention: propagate full posterior (vector of responsibilities over keys)
- ReLU: collapse to posterior mode (degenerate)

These aren't three different mechanisms. They're three different choices of how much posterior information to keep.

**One clarification needed:**

> "Linear activations keep everything (no collapse)"

But linear layers don't have hypothesis structure. It's not that linear keeps the full posterior—it's that linear never induced a posterior in the first place. Linear is *outside* the framework, not a special case within it.

Maybe: "Linear activations bypass the hypothesis structure entirely; the implicit EM framework does not apply."

**The "information bottleneck" framing:**

This is the bridge to a whole other literature. Tishby's information bottleneck work. Could be a connection or a distraction—your call whether to gesture at it.