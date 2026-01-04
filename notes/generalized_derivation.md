# Generalized Activation Synthesis

## Goal

Given a **kernel** that maps distance-like representations to unnormalized likelihoods, derive the **activation function** that is consistent with implicit EM.

Here “consistent” means:

1. responsibilities are induced by exponentiation + normalization, and
2. the activation is a posterior statistic computed from those responsibilities.

The activation is not chosen. It is derived.

---

## Objects and Notation

* Input: (x)

* Linear (or distance-producing) map:
  [
  z = f_\theta(x)
  ]
  In the simplest case, (z = Wx + b). In distance-based models, coordinates of (z) are interpreted as signed Mahalanobis-like deviations.

* Latent hypothesis index: (h)
  This can represent:

  * mixture component (j),
  * sign (s\in{\pm1}),
  * prototype + sign ((j,s)),
  * or any discrete set of competing explanations.

* Energy (distance) function induced by a kernel:
  [
  d_h(z) ;; \text{(lower is better)}
  ]

* Unnormalized likelihood (kernel likelihood):
  [
  P_h(z) = \exp(-d_h(z))
  ]

---

## The EM-Preserving Requirement

A kernel preserves the implicit EM property when it participates in a **normalized exponential family** across alternatives:

### Normalization across hypotheses

[
Z(z) = \sum_h \exp(-d_h(z))
]

### Responsibilities (posterior over hypotheses)

[
r_h(z) = \frac{\exp(-d_h(z))}{\sum_{h'} \exp(-d_{h'}(z))}
]

These responsibilities define the implicit E-step: a soft assignment of the current representation to competing hypotheses.

### Log-sum-exp objective

The canonical objective that induces these responsibilities is:
[
L(z) = \log \sum_h \exp(-d_h(z)) = \log Z(z)
]

This structure is what guarantees EM-like learning dynamics: responsibilities arise from normalization, and responsibility-weighted updates arise from gradient flow.

---

## What “Deriving the Activation” Means

An **activation** is a deterministic function of (z) that returns a transformed value (a(z)) used by subsequent layers.

In the implicit EM perspective, the activation is a **posterior statistic** computed from (r_h(z)). The statistic depends on what latent quantity the hypothesis index (h) represents.

### General form

Choose a statistic (g(h)) (scalar, vector, or structured), and define:
[
a(z) = \mathbb{E}[g(h)\mid z] = \sum_h r_h(z), g(h)
]

This is the central template:

* kernel defines energies (d_h(z)),
* energies define responsibilities (r_h(z)),
* responsibilities define an activation via a posterior expectation.

---

## Step-by-Step Recipe

### Step 1: Identify the representation (z)

Specify the quantity the layer produces before nonlinearity:
[
z = f_\theta(x)
]
Typical case: a signed Mahalanobis coordinate or distance-like feature.

### Step 2: Choose the hypothesis set (\mathcal{H})

Decide what alternatives compete under normalization.

Examples:

* Sign hypotheses: (\mathcal{H}={+1,-1})
* Prototype hypotheses: (\mathcal{H}={1,\dots,K})
* Prototype + sign: (\mathcal{H}={1,\dots,K}\times{+1,-1})

This choice determines what “assignment” means.

### Step 3: Specify the kernel via energies (d_h(z))

Define a distance / energy per hypothesis.

Requirements:

* (d_h(z)) is finite and (almost everywhere) differentiable in (z)
* lower energy corresponds to higher likelihood

Examples:

* Gaussian: (d_h(z)=\frac{|z-\mu_h|^2}{2\sigma^2})
* Laplace: (d_h(z)=\frac{|z-\mu_h|_1}{b})
* Student-t: (d_h(z)=\log(1+|z-\mu_h|^2/(\nu\sigma^2)))

### Step 4: Form responsibilities by normalized exponentials

Compute
[
r_h(z) = \frac{\exp(-d_h(z))}{\sum_{h'} \exp(-d_{h'}(z))}
]

This is the implicit E-step.

### Step 5: Choose the posterior statistic that defines the activation

Select (g(h)) according to what you want the activation to represent.

Common choices:

* Posterior mean of a signed variable: (g(s)=s)
* Posterior mean of a prototype parameter: (g(j)=\mu_j)
* Responsibility vector itself: (g(h)=\mathbf{e}_h) (one-hot basis)
* Posterior expected value contribution: (g(h)=v_h) (attention-style)

Then define:
[
a(z) = \sum_h r_h(z), g(h)
]

### Step 6: Simplify to obtain the closed-form activation (if possible)

For some kernels, the posterior expectation yields a known nonlinearity:

* symmetric Gaussian sign model ⇒ tanh
* multiway competition ⇒ softmax / softmin
* other kernels ⇒ other smooth saturating functions

If no closed form exists, the activation is still well-defined by the expectation.

---

## Why This Preserves Implicit EM

The EM structure does not come from the activation itself. It comes from the **log-sum-exp normalization over energies**.

As long as the kernel is used inside:
[
L(z) = \log \sum_h \exp(-d_h(z)),
]
responsibilities exist and satisfy:

* (r_h(z)\ge 0)
* (\sum_h r_h(z)=1)

This induces:

* competition between hypotheses,
* soft assignment,
* responsibility-weighted learning signals.

The activation is then a readout of this assignment state.

---

## Design Degrees of Freedom

This framework exposes three independent choices:

1. **Representation map** (z=f_\theta(x))
   Determines what geometry the model can express (e.g. Mahalanobis axes).

2. **Kernel / energy family** (d_h(z))
   Encodes noise assumptions, robustness, and saturation behavior.

3. **Posterior statistic** (g(h))
   Determines what is passed forward: mean sign, mean prototype, mixture weights, etc.

The activation is determined by (2) and (3), given (1).

---

## Minimal Checklist for an EM-Compatible Activation Derivation

A kernel supports a clean activation derivation in this framework if:

* It defines energies (d_h(z)) for a discrete hypothesis set (\mathcal{H}).
* Likelihoods are of the form (\exp(-d_h(z))).
* Hypotheses are normalized via a log-sum-exp / softmax across (h).
* The activation is defined as a posterior statistic (\sum_h r_h(z)g(h)).

If any of these are removed (especially normalization), responsibilities cease to exist and the implicit EM interpretation no longer applies.
