# Paper Outline: Volume Control Transfers, Conditioning Does Not

**Working title (recommended): Volume Control Transfers, Conditioning Does Not:
The Locality of Implicit EM in Neural Networks.**

Center of gravity: implicit EM's optimization conditioning is *local, graded, and
structural*. Volume control transfers to intermediate sites; the optimization
conditioning does not survive ordinary backpropagation through learned dense maps.
The recommended title is the most information-dense candidate and states the
one-sentence contribution directly; commit to it early because it fixes the shape
of the abstract and introduction.

## Title Candidates (recommendation first)

- **Volume Control Transfers, Conditioning Does Not** (recommended — states the result)
- Where Implicit EM Lives
- The Boundary of Implicit EM
- Locality of Implicit EM in Neural Networks
- When Responsibility-Weighted Gradients Survive Backpropagation
- Backpropagation as Parameter Sharing: A Failure Mode of Implicit EM

## Status of this outline (revised)

This revision folds in structural gaps a reviewer would flag and resolves the
open decisions the proposal left hedged. Changes from the prior version:

- §2 is now a real **Related Work / Positioning** section (was dispersed into the
  discussion).
- §3 (Theory) is restructured around **numbered Propositions with explicit
  hypotheses and a stated non-necessity guardrail** (was inline `text` blocks).
- New **§0 Notation and Conventions** block fixes the distance/energy sign
  convention once (the prior draft flagged this as unresolved).
- New **Predictions ledger** (§7.0) — prediction → measured → verdict — makes the
  causal claims auditable and preempts "post-hoc storytelling."
- **Threats to validity** split out from scope-Limitations (§9).
- The **λ-resolution curve** and **basin/LMC analysis** are promoted from
  "optional" to main results, matching the proposal's own ranking.
- New **§8.6 Relation to the attention project** — the companion paper
  *Depth Is Not Temperature* (E:\Projects\attention_collapse) explicitly invokes
  this paper's locality result as "unpublished where-EM-lives results." This paper
  is the controlled proof of the claim that paper uses in the wild. This fixes the
  stochastic-map decision (see §7.6 and the decision note at the end).
- Scope note: the user has local GPU (RTX 3080 Ti, 12 GB + shared). All remaining
  experiments (Adam arm, λ-curve, basin, depth sweep, stochastic-map,
  Fashion-MNIST) are feasible; this outline assumes the submission-bar items
  (proposal 1–4) are run and treats 5–7 as strengtheners with an explicit
  in/out decision at the end.

## 0. Notation and Conventions

State once, use everywhere. A single table prevents the sign-convention drift the
prior draft flagged ("be consistent in final draft").

- **Convention: energies/distances, lower = better match.** `d_j` is the energy of
  component `j`; small `d_j` means component `j` explains the input well.
- `r = softmax(-d)` — responsibilities (posterior over components). `sum_j r_j = 1`.
- `L_LSE(d) = -log sum_j exp(-d_j)`; the identity is `dL_LSE/dd_j = r_j`
  (Paper 1). Sign: differentiating the *negative* log marginal gives `+r_j`.
- `p = softmax(h)` — the output-layer class posterior (distinct from `r`); `1[j=y]`
  is the label clamp.
- `W1` (input→components, private per-component rows), `W2` (components→classes,
  the *shared* learned dense map — the "corridor").
- `NLS(d)_j = d_j + log sum_k exp(-d_k)` — NegLogSoftmin; distance-preserving
  calibration, `exp(-NLS(d)_j) = r_j`.
- Volume control: `L_var` (anti-collapse), `L_tc` (anti-redundancy), `lambda` the
  overall auxiliary weight, `lambda_reg = 0.001` unless noted.
- `sigma_max(W2)` — largest singular value of the corridor map.
- The "½-saturation diagnostic": distance of `||H_d L_LSE||_2` from `1/2`.

## Abstract Skeleton

Gradient descent on log-sum-exp objectives produces responsibility-weighted
gradients, so standard neural objectives can instantiate generalized EM without
explicit latent-variable updates. But this identity is local: it holds at the
objective interface, and does not imply that a whole deep network inherits EM's
optimization properties. We characterize this boundary. The Hessian of an LSE
objective in energy coordinates is uniformly bounded by 1/2; with private
component parameters this yields parameter-independent curvature bounded by a
data constant. When the same site is composed behind a learned dense map, the
curvature contains a conjugated softmax Hessian, W^T(diag p - pp^T)W, and its
upper bound scales with the learned spectrum sigma_max(W2)^2 — the
parameter-independent guarantee is forfeited (we do not claim the bound proves
curvature is large; the measurements do that). Composition thus removes the
uniform guarantee, and empirically the composed curvature is 13-100x larger,
non-monotone, and parameter-dependent.

Experiments in supervised MNIST networks confirm the split. Volume control
applied at an intermediate EM site eliminates dead units and redundancy across
widths, showing that local collapse failures transfer. But the optimization
signatures observed in the single-layer implicit-EM model disappear under
ordinary supervised composition. Measuring gradients shows that the
cross-entropy path dominates the local EM path by 30-70x and is nearly
orthogonal to it. Cutting that path with stop-gradient restores learning-rate
insensitivity and convergence-time invariance; increasing the EM weight produces
a continuous transition. The result is a local, graded, structural account of
implicit EM: volume control repairs failures at EM sites, but EM conditioning
survives only when responsibility gradients reach private parameters through
simplex-compatible operations.

## 1. Introduction

### 1.1 Starting Point

Paper 1 established the algebraic identity:

```text
For LSE objectives over distances or energies, dL/dd_j = -r_j.
```

Responsibilities arise as gradients. Standard neural objectives therefore
instantiate implicit generalized EM at the objective interface.

Paper 2 showed that this is prescriptive in an unsupervised model:

- LSE + volume control produced structured prototype-like features.
- Removing volume control produced the predicted failure modes.
- The single-layer model had unusual optimization behavior: SGD learning-rate
  insensitivity, little/no Adam advantage, and roughly fixed convergence time.

### 1.2 The Temptation

Two tempting but wrong inferences follow from the identity, and this paper is the
boundary that rules both out.

- **Temptation 1 (inheritance).** That whole neural networks inherit EM's
  conditioning because responsibility gradients flow through backpropagation.
- **Temptation 2 (depth-as-temperature).** That stacking LSE/softmax sites makes
  *depth itself* act as an annealing temperature schedule — an EM property
  distributed across layers. This is Temptation 1 sharpened into a specific,
  falsifiable corollary: a distributed schedule is exactly an EM property
  propagating through composition.

Our locality result forbids both. Because conditioning does not survive
composition (§3.3), there is no distributed schedule to find. The
depth-as-temperature corollary is tested directly — and falsified — in follow-up
work on transformers (§8.6); we present it here as a prediction the theory makes,
not a hypothesis the paper endorses.

This paper asks the boundary question:

Under what conditions does the implicit-EM optimization property hold, and what
destroys it?

### 1.3 Main Claim

Implicit EM is local, graded, and structural.

- Local: it exists at exponentiation + normalization sites and extends zero
  learned dense layers beyond them.
- Graded: in a real network, behavior varies continuously with the measured
  share of EM-structured gradient reaching the parameters.
- Structural: failures split into at-site failures and structural failures.
  At-site failures are repaired locally; structural failures require isolation,
  dominance, or simplex-compatible transport.

Key sentence:

Backpropagation composition is parameter sharing in disguise. The neural failure
is the classical EM failure that appears when components no longer own private
parameters and the M-step stops separating.

**Reconciliation (critical — state this explicitly; §1.3, §3.5, §7 must agree).**
The punchline and the centerpiece result (λ=1 joint ≈ stop-gradient: dominance,
not connectivity) are not in tension once the mechanism is split into two
questions:

- *Why is the foreign contribution ill-conditioned?* — **Structure.** Sharing W1
  across all upper hypotheses through W2 makes the CE-path M-step non-separable
  (Prop 3), so the CE-path gradient at W1 carries the `σ_max(W2)²` curvature
  (Prop 2). This is "parameter sharing in disguise." It is always true when the
  corridor is connected.
- *When does that contribution decide the landscape?* — **Dominance.** The gradient
  arriving at W1 is a sum of a well-conditioned EM term (Prop 1) and the
  ill-conditioned CE-path term. Whichever dominates sets the effective curvature.
  λ=1 makes EM dominate → conditioning survives *despite* full sharing;
  λ=0.001 makes the CE path dominate → conditioning is destroyed; stop-gradient
  zeroes the CE term → survives trivially.

So sharing does not by itself destroy conditioning; it *manufactures an
ill-conditioned contribution*, and dominance decides whether that contribution
wins. Proposition 3's clean classical story is the **limiting case** (CE path
dominant, EM path absent), not the whole mechanism. This is exactly what makes the
claim *graded* rather than binary — and it is why the λ-resolution curve (§7.6),
not the stop-gradient point alone, is the central figure.

### 1.4 Contributions

1. Prove the conditioning contrast between private-parameter LSE objectives and
   composed dense maps.
2. Classify operations that preserve or destroy the responsibility-gradient
   invariant.
3. Introduce the at-site / structural failure taxonomy.
4. Show that volume control transfers to intermediate supervised layers and
   repairs dead/redundant units.
5. Show that optimization conditioning does not survive ordinary composition.
6. Establish causally that conditioning is restored by stop-gradient or EM
   dominance.
7. Introduce the 1/2-saturation diagnostic for mixture sharpness.

## 2. Related Work and Positioning

A dedicated section (not scattered rebuttals in the discussion). A ~14-page paper
carrying both theory and experiment needs its positioning stated up front. Group
the citations into four claims-of-relatedness:

**EM as an optimization object.**
- Dempster, Laird & Rubin (1977): EM and the missing-information principle; the
  rate of convergence is governed by the fraction of missing information — the
  classical antecedent of our "diffuse responsibilities converge slowly" mode.
- Neal & Hinton (1998): the free-energy / coordinate-ascent view that makes
  "gradient step = EM step" legible.
- Xu & Jordan (1996): EM as a preconditioned gradient method with a
  data-and-model-dependent conditioning — the closest prior statement that EM's
  *conditioning* is structural, which we make precise and then break by composition.
- Böhning (1992) and the Popoviciu variance inequality: the source of the uniform
  `1/2` bound on the multinomial-logistic Hessian. Our Proposition 1 is a direct
  application; the novelty is the composition contrast (Proposition 2), not the bound.

**Classical EM failure modes.** Shared-parameter / non-separable-M-step EM and
generalized EM (GEM). The paper's punchline — backprop composition is parameter
sharing in disguise — is that the neural failure *is* this classical one. Cite the
standard treatments of coupled M-steps and singular-component collapse.

**Generative vs discriminative training.** Ng & Jordan (2002) is the high-level
phenomenon a skeptic will reach for ("this is just gen-vs-disc"). Position early
and explicitly: we cite it as the phenomenon and contribute the *mechanism inside
one architecture* (curvature dichotomy, measured gradient dominance, causal
restoration). See §8.3.

**Loss-landscape / basin structure.** Linear mode connectivity (Frankle et al.
2020; Entezari et al. 2022 for permutation alignment). Used for the basin analysis
(§7.6 / §8.4) that formalizes the fine-tuning observation. Hungarian/permutation
matching is required or the barrier is a relabeling artifact.

**Gradient conflict / multi-task and auxiliary-loss balancing.** We measure
dominance ratios and cosines between a main (CE) and auxiliary (EM) loss at a
shared parameter — squarely the multi-task / gradient-interference literature.
Cite PCGrad / gradient surgery (Yu et al. 2020), auxiliary-task and loss-weighting
work (e.g. GradNorm, Chen et al. 2018), and **gradient starvation** (Pezeshki et
al. 2021, the closest in spirit — one signal drowning another). Our contribution
relative to this line: we do not propose a balancing method; we show the ratio is
the *causal control variable* for an interpretable structural property
(EM conditioning), with a stop-gradient positive control. Note our finding that the
two gradients are near-**orthogonal** (drowned, not opposed) distinguishes this
from the conflict regime PCGrad targets.

**Local / decoupled learning (beyond DBNs).** The design rule "create EM sites
locally, protect with stop-gradients" sits next to greedy layer-wise pretraining /
DBNs (Hinton et al. 2006), decoupled greedy layer-wise training (Belilovsky et al.
2019), synthetic gradients / decoupled neural interfaces (Jaderberg et al. 2017),
and Forward-Forward (Hinton 2022). Position P004 (layer-wise EM) as the constructive
paper in this family; P001 supplies the *why* (owned sites vs. foreign-gradient
corridors).

**The framework's own lineage.** Paper 1 (implicit EM identity, arXiv:2512.24780),
Paper 2 (decoder-free volume-control model, arXiv:2601.06478), and the companion
Mahalanobis paper (arXiv:2410.19352). **The companion attention paper** (*Depth Is
Not Temperature*) is positioned in §8.6: it invokes this paper's locality result on
real transformers; this paper is its controlled proof.

**What is deliberately not claimed as related.** No general deep-network
conditioning theory (Hessian-based generalization work, NTK, etc.) — the claims are
about EM sites and their corridors only. Mention once to bound scope.

## 2.5 Background: Where EM Lives

(Numbered 2.5 to avoid the §2/§2b collision; in the final manuscript, promote
Related Work to §2 and this Background to §3, cascading the later section numbers.
Kept as 2.5 here so the many cross-references below — §3.x theory, §7.x results —
stay stable during outlining.)

### 2.1 Gradient-Responsibility Identity

State the LSE identity briefly and connect it to Paper 1.

For distances:

```text
L(d) = -log sum_j exp(-d_j)
r = softmax(-d)
dL/dd_j = r_j
```

Sign convention depends on whether the paper writes energies, distances, or
negative distances. Be consistent in final draft.

For cross-entropy classification:

```text
L = -z_y + log sum_k exp(z_k)
dL/dz_j = softmax(z)_j - 1[j = y]
```

The output softmax is a label-clamped responsibility interface.

### 2.2 Locality

The identity applies where the loss sees distances/energies. It does not make
all earlier hidden layers into mixture models.

Correct statement:

The loss geometry imposes EM structure at the output, and gradients from this
structure flow backward to shape internal representations.

Incorrect statement:

The whole network is doing EM.

### 2.3 Volume Control

Recall Paper 2:

- LSE creates competition / responsibility structure.
- Variance prevents collapse/dead components.
- Decorrelation prevents redundancy.
- Together these play the neural role of the log-determinant in mixture models.

This section should be compact. The new paper is not re-proving Paper 2.

## 3. Theory: The Conditioning Boundary

This section should carry the core mathematical claim.

Write §3 as numbered environments, not prose. Each Proposition states
hypotheses, the claim, and a one-line proof idea; full proofs go to an appendix.
A boxed **Scope of the claim** note after Proposition 2 states what is *not*
proved (this is the guardrail for the top-ranked "iff temptation" risk).

### 3.1 The Simplex Invariant (Lemma 1)

**Lemma 1.** For `L_LSE(d) = -log sum_j exp(-d_j)` with `r = softmax(-d)`:
the gradient `grad_d L = r` is a conditional expectation on the simplex
(nonnegative, sums to one, per-component aligned), and the Hessian is the
covariance of a categorical distribution,

```text
H = diag(r) - rr^T,    0 <= H,    ||H||_2 <= 1/2.
```

*Proof idea.* Böhning's bound / Popoviciu's inequality on the variance of a
`[0,1]`-bounded categorical. The bound is **uniform in `d`** — it holds at every
point of training, independent of parameters.

### 3.2 Private Parameters Inherit the Bound (Proposition 1)

**Proposition 1 (conditioning under private energies).** Let each component energy
be a private affine map `d_j = w_j^T x + b_j` (each component owns its parameters;
no sharing). Then in parameter coordinates,

```text
||grad_theta L|| <= ||x||,    ||H_theta L||_2 <= (1/2)||x||^2,
```

and under batch-mean training `||H||_2 <= (1/2) E||x||^2`, a **data constant
independent of the current parameters**.

*Consequence (explains the Paper 2 anomaly).* The admissible learning-rate range
does not shrink as parameters move; curvature is not learned or anisotropic; an
adaptive optimizer has little to correct. This is *why* SGD was learning-rate
insensitive, Adam gave no advantage, and convergence time was fixed.

*Upgrade to a quantitative prediction (near-free — do it).* The bound
`L = (1/2) E||x||^2` is a *checkable number*. Smooth-descent theory gives a stable
SGD step size up to ≈ `2/L`. Compute `E||x||^2` on MNIST, report the predicted
admissible-lr ceiling, and overlay the measured lr tolerance from the
stop-gradient sweep (§7.2). If they match, "explains Paper 2's anomaly" upgrades
from qualitative to **quantitative** for essentially the cost of one expectation and
one overlay line on an existing figure. High value-to-effort; flag as a headline
supporting result.

*Extension to smooth monotone kernels (Proposition 1').* For `d_j = phi(w_j^T x + b_j)`
with `phi` monotone and `|phi'|, |phi''|` bounded (Softplus), the chain rule adds
bounded diagonal and second-order factors; parameter-independence survives up to
constants. **ReLU:** the bounds hold almost everywhere, with a zeroth-order
absorbing-state caveat (a dead unit's zero gradient can become permanent) — stated
as a caveat, not swept under Proposition 1'.

### 3.3 Composition Forfeits the Uniform Bound (Proposition 2)

Name it "**forfeits the uniform bound**," not "loss of conditioning." The
proposition is an upper-bound statement; it shows the parameter-independent
*guarantee* is no longer available, not that curvature is necessarily large. The
"and in practice it really is large" weight is carried by §7.4 (measured 13-100x,
non-monotone). Keep this split sharp — a theory reviewer pokes exactly here.

**Proposition 2 (composition forfeits the uniform bound).** Compose the same EM
site behind a learned dense map: `y = NLS(d)`, `h = W2 y`, `L_CE = CE(h, c)`,
`p = softmax(h)`. Then the Hessian of `L_CE` with respect to `d` contains the
conjugated softmax Hessian

```text
J^T W2^T (diag(p) - pp^T) W2 J   (+ curvature terms from NLS),
```

so that `||H_d L_CE||_2 <= O(sigma_max(W2)^2)`. The uniform `1/2` bound of Lemma 1
no longer holds; the controlling quantity is now **learned, time-varying, and
anisotropic**. Whether it is actually large is an empirical question, answered
affirmatively in §7.4.

**Scope of the claim (boxed).** Propositions 1–2 establish *sufficient* conditions
for preservation (private parameters, monotone kernels, unmixed objective) and
*demonstrate* loss of conditioning when a shared sign-indefinite map intervenes.
They do **not** prove necessity in general architectures, and give no convergence
proof for the composed system. The paper says "sufficient conditions + demonstrated
failure under each violated condition," never "iff." (Top-ranked risk from the
proposal; the guardrail lives here.)

**LayerNorm.** State the clean theorem *without* LayerNorm and note that the
empirical curvature measurements (§7.4) are on the actual model including LayerNorm,
so the measured constants absorb its effect. Alternatively fold its (bounded,
per-sample) Jacobian into the constants — decide in drafting; the cleaner theorem
without it is preferred.

### 3.4 Operations That Preserve or Destroy the Invariant

Preserve:

- private per-component parameters
- per-coordinate monotone maps
- bounded smooth kernels, e.g. Softplus
- stochastic/Markov maps, at least first order

Destroy:

- dense sign-indefinite learned maps
- objective mixing
- hard gates
- shared lower parameters

ReLU:

State as a.e. compatible with the curvature bound but with a zeroth-order
absorbing-state pathology: zero gradient can become permanent.

### 3.5 Classical EM Connection (derive it, don't just tabulate)

This is the paper's emotional center ("parameter sharing in disguise"). It must be
a short *argument*, not a table row, or the punchline reads as analogy.

**The derivation (Proposition 3, informal).** Classical EM's efficiency comes from
the E-step making the expected complete-data objective `Q` *separate* over
components: with private component parameters `theta_j`, `Q = sum_j Q_j(theta_j)`
and the M-step decouples into independent per-component problems, each convex for
exponential-family components. Now share a lower map: let every component's energy
depend on common parameters `W1` through `W2` (the corridor). Then `Q` no longer
separates — `d^2 Q / dW1_a dW1_b != 0` across components — and the M-step is a
single coupled non-convex problem. This is *exactly* the classical
shared-parameter / non-separable-M-step failure. The conjugated Hessian of
Proposition 2 is the coordinate-space fingerprint of the same coupling.

**Then the table (as a map, after the argument).**

| Classical EM failure | Why EM fails | Neural incarnation | Where in this paper |
|---|---|---|---|
| Shared component parameters | `Q` stops separating; M-step coupled | Backprop through shared `W1`/`W2` | §3.5, Prop. 2 |
| Non-separable / generalized EM | No closed-form M-step; nonconvex inner problem | Component energies via deep maps | §3.5 |
| Singular components | Variance → 0, unbounded likelihood | LSE-only collapse | §5.1 (Exp 1) |
| Hard assignment | Zero-responsibility units cannot recover | Dead ReLU units | §5.1–5.2 |
| High missing information | Uniform responsibilities → slow convergence | Diffuse responsibilities; low ½-saturation | §7.4 |

Key claim: EM works cleanly when the E-step separates the objective over private
component parameters. Backpropagation composition violates this by making lower
parameters shared across all upper hypotheses. The neural failure is not new — it
is the classical failure rediscovered inside the chain rule.

**Scope of Prop 3 relative to the graded result (must state — see §1.3
reconciliation).** Proposition 3 explains why the *CE-path contribution* to the
gradient at W1 is ill-conditioned: sharing makes its M-step non-separable. It does
**not** say the composed system is always ill-conditioned — the λ=1 joint arm shares
parameters fully yet stays well-conditioned because the EM contribution dominates
the sum (§7.2–7.3). Prop 3 is therefore the *structural* half (why the foreign term
is bad); the graded result is the *dominance* half (when it wins). The clean
classical picture is the limiting case where the CE path dominates and the EM path
is absent. Do not let this section imply sharing alone destroys conditioning; that
is precisely the sentence a reviewer would quote against the λ=1 result.

## 4. Experimental Setup

### 4.1 Model

Small supervised MNIST classifier:

```text
x -> Linear(784, K) -> ReLU -> d
  -> optional NegLogSoftmin / volume-control site
  -> Linear(K, 10) -> LayerNorm -> CE
```

For the causal experiment:

```text
aux(d) = LSE(d) + variance penalty + decorrelation penalty
```

NegLogSoftmin:

```text
y_j = d_j + log sum_k exp(-d_k)
exp(-y_j) = r_j
```

Explain it as distance-preserving calibration that embeds the LSE partition
function and gives the intermediate site a competitive Jacobian.

### 4.2 Metrics

Representation health:

- dead units
- minimum variance
- redundancy
- responsibility entropy

Optimization conditioning:

- accuracy/probe spread across learning rates
- convergence-time variability
- Adam vs SGD gap
- gradient norm ratio
- cosine between CE and EM gradients
- Hessian spectral norms with respect to d

### 4.3 Experimental Logic

The experiments follow the taxonomy:

- Experiments 1-2 test at-site failures and volume-control transfer.
- Experiment 3 observes structural failure of conditioning under composition.
- Experiment 4 measures the mechanism and causally restores the property.

## 5. Results I: At-Site Failures Are Repaired Locally

Use Experiments 1 and 2 here.

### 5.1 Volume-Control Ablation

Design:

- MNIST
- K = 25
- 50 epochs
- Adam lr = 0.001
- lambda = 0.001
- 10 seeds

Configs:

| Config | NLS | Var | Decorr |
|---|---:|---:|---:|
| Baseline | no | no | no |
| NLS only | yes | no | no |
| NLS + Var | yes | yes | no |
| NLS + Var + Decorr | yes | yes | yes |
| Var + Decorr only | no | yes | yes |

Headline numbers:

| Config | Dead Units | Min Var | Redundancy | Resp Entropy | Accuracy |
|---|---:|---:|---:|---:|---:|
| Baseline | 1.40 +/- 0.49 | 0.000 | 20.5 | 3.08 | 96.09% |
| NLS only | 1.20 +/- 0.98 | 0.264 | 24.9 | 2.93 | 96.14% |
| NLS + Var | 0.00 | 11.74 | 187.0 | 2.09 | 95.75% |
| NLS + Var + Decorr | 0.00 | 7.84 | 13.9 | 2.46 | 96.34% |
| Var + Decorr only | 0.10 | 3.55 | 13.3 | 2.65 | 96.40% |

Interpretation:

- Supervised gradients do not provide intermediate volume control.
- NLS without volume control is insufficient.
- Variance alone is pathological: alive but redundant units.
- Full volume control repairs the site.
- Var + Decorr without NLS is useful regularization but not the same calibrated
  EM structure.

This is the transfer result: collapse and redundancy are local failures, so
local volume control fixes them.

### 5.2 Capacity Sweep

Design:

- baseline vs NLS + Var + Decorr
- K in {16, 25, 36, 49, 64}
- 5 seeds

Main claims:

- Baseline has dead units at every width, about 5-7% dead fraction.
- Volume control eliminates dead units at every width.
- Redundancy grows with width; volume control lowers the slope and tightens
  seed variability.
- Accuracy benefit is capacity/lambda dependent and should not be oversold.

Use this to make the paper honest:

The structural health claim is stronger than the accuracy claim. This paper is
not a benchmark paper.

## 6. Results II: Conditioning Does Not Survive Composition

Use Experiment 3.

### 6.1 Optimizer Sweep

Design:

- NLS + Var + Decorr
- SGD and Adam
- lr in {0.0001, 0.001, 0.01, 0.1}
- K = 25
- lambda = 0.001
- 3 seeds

Headline:

- SGD spans 83.83% to 96.23% accuracy.
- Adam at 0.0001 reaches 95.34%; Adam at 0.001 reaches 96.38%.
- Adam clearly helps, unlike Paper 2.

Interpretation:

The EM-conditioned local site exists, but the gradient reaching W1 is dominated
by the composed supervised path. The result is ordinary deep-network
conditioning.

### 6.2 Loss-Feature Decoupling

At high Adam learning rates:

- regularization loss decreases strongly
- min variance explodes, e.g. 423 and 30628
- accuracy barely changes

This partially echoes Paper 2: the auxiliary geometry has degrees of freedom
orthogonal to task accuracy.

### 6.3 Transition to Mechanism

Report 3 originally inferred the EM path was downweighted by lambda alone. That
was too crude. Experiment 4 measures the actual gradient ratio and shows the
right mechanism:

- 30-70x CE dominance at lambda = 0.001
- near-zero cosine
- domination, not opposition

## 7. Results III: The Mechanism and Causal Test

This is the centerpiece.

### 7.0 Predictions Ledger (pre-empt "post-hoc storytelling")

Every experiment made predictions before it ran. Show them against outcomes in one
auditable table. This is cheap and disproportionately convincing for a
causal/boundary paper.

| # | Prediction (stated in advance) | Measured | Verdict |
|---|---|---|---|
| Exp1 | Supervision does not give intermediate volume control; baseline has dead units | min-var = 0, ≥1 dead ReLU every seed | ✓ |
| Exp1 | Partial VC (var only) worse than none | redundancy 20→187; accuracy worst | ✓ |
| Exp3 | Paper 2 optimizer signature disappears under composition | SGD spans 83.8–96.2%; Adam ≫ SGD | ✓ |
| Exp4-P1 | CE gradient dominates EM gradient at small λ, near-orthogonal | 30–70×, cos ≈ 0 | ✓ |
| Exp4-P2 | LSE Hessian ≤ ½ always; CE-path scales with `sigma_max(W2)` | ½ held; CE-path 13–100× larger | ✓ |
| Exp4-P3 | Stop-gradient restores learning-rate insensitivity inside supervised net | spread 10.9→1.3 pts | ✓ |
| Exp4 | λ=1 joint ≈ stop-gradient (dominance, not connectivity) | indistinguishable | ✓ |
| Exp4 | Restored conditioning costs task alignment | 88% vs 96% probe | ✓ (honest cost) |

Also state the tests that *could have failed and half did*, in the spirit of the
companion paper's discriminator discipline: e.g. λ itself was predicted (Report 3)
to equal the gradient ratio (~1000×) and did **not** — the measured ratio is
30–70×. Keep the mechanism, correct the magnitude, and log the correction visibly.

### 7.1 Experiment 4 Design

Arms:

| Arm | Total loss | CE reaches W1? | Expected regime |
|---|---|---:|---|
| joint, lambda = 0.001 | CE + 0.001 aux | yes | CE-dominated |
| joint, lambda = 0.03 | CE + 0.03 aux | yes | balanced |
| joint, lambda = 1 | CE + aux | yes | EM-dominated |
| stop-gradient | CE(head(y.detach())) + aux | no | pure EM at W1 |

Feature quality:

Train a fixed-protocol linear probe on frozen distances so the measurement is
decoupled from how well the supervised head trained.

### 7.2 Causal Restoration of Conditioning

Probe accuracy spread across 1000x SGD learning-rate range:

| Arm | Spread |
|---|---:|
| joint, lambda = 0.001 | 10.9 pts |
| joint, lambda = 0.03 | 5.9 pts |
| joint, lambda = 1 | 1.2 pts |
| stop-gradient | 1.3 pts |

Key result:

Cutting the CE-to-W1 path restores Paper 2's learning-rate insensitivity inside
a supervised network. Supervision was not the problem; the corridor was.

Convergence-time invariance also returns:

- stop-gradient reaches 95% final LSE around 22-32 epochs across learning rates
- joint lambda = 0.001 is erratic and sometimes never reaches the threshold

**Frame the convergence-time metric carefully.** "Epoch to reach 95% of final LSE"
is a natural stability measure in the EM-dominated arms, but in the CE-dominated
arm *nobody is optimizing LSE* — the auxiliary term is drowned — so "erratic /
never converges" there means "LSE is along for the ride," not "optimization
failed." Report it as *the EM site never stabilizes when the CE path dominates*,
which is the intended reading, and do not imply the CE-dominated model failed to
train (its accuracy is fine; §6.1). Consider reporting the metric only for the
arms where LSE is a live objective, with the CE-dominated arm shown as a contrast.

### 7.3 Gradient Competition

Measured at the distance interface:

| Arm | CE / weighted aux | Cosine |
|---|---:|---:|
| joint, lambda = 0.001 | 30-70x | approx 0 |
| joint, lambda = 1 | 0.02-0.05x | approx 0 |

Interpretation:

The EM signal is drowned, not opposed. The two gradients are nearly orthogonal.
Lambda does not directly equal the gradient ratio because the raw auxiliary
gradient is larger.

The lambda = 1 joint arm behaves like stop-gradient, proving that conditioning
follows gradient dominance, not graph connectivity.

**Two measurement fixes (do before drafting the figure):**
- *Cosine vs chance, not vs zero.* In K=25 the cosine of two random vectors has
  std ≈ 1/√K ≈ 0.2. "Cosine ≈ 0" is the high-dimensional default and reads as
  vacuous unless stated relative to that noise floor. Report the measured cosine
  distribution against the ±0.2 chance band (and how many σ below it, if any), or
  the orthogonality claim is unconvincing. If the measured cosine is *within* the
  chance band, the honest statement is "no detectable alignment," which still
  supports "drowned, not opposed."
- *Optimizer mismatch.* The gradient-competition numbers were taken during **Adam**
  training at lr=0.001, while the conditioning sweep (§7.2) is **SGD**. Either add
  an SGD gradient-ratio measurement so the mechanism and the causal sweep are on the
  same optimizer, or state the mismatch explicitly and argue the ratio is an
  optimizer-independent property of the loss geometry (preferable to measure, not
  assert).

### 7.4 Curvature Dichotomy

Measurements:

- LSE Hessian with respect to d is always <= 1/2.
- Pure EM / stop-gradient saturates the bound: 0.4993-0.4997.
- CE-path Hessian is roughly 13-100x larger, non-monotone, and consistent with
  W2 spectral scaling.

Main point:

The 1/2 bound is not just a worst-case guarantee. Under healthy EM training, the
mixture sharpens until boundaries form two-way ties, which saturate the bound.
Distance from 1/2 becomes a diagnostic of mixture sharpness.

### 7.5 The Trade-Off

EM-dominated features probe around 88%; CE-dominated features probe around 96%.

This should be stated plainly:

Well-conditioned optimization of a less task-aligned objective does not beat
ill-conditioned optimization of the task objective. The EM regime buys
optimization stability, not accuracy.

**Pre-empt the deflationary reading (do this explicitly).** In the stop-gradient
arm W1 never sees label information, so a skeptic reads 88-vs-96 as "unsupervised
features probe worse than supervised — obviously." Two defenses:

- *It is a continuum, not a binary.* The λ-resolution curve (§7.6) shows the
  probe accuracy and conditioning spread trade off smoothly along the *measured*
  gradient ratio. This is the primary rebuttal and a second reason §7.6 is central.
- *Name who wants conditioning.* Conditioning is worth buying when the cost is paid
  somewhere other than output accuracy: learning-rate robustness with **no tuning**,
  seed-to-seed **reproducibility**, **fixed convergence time**, and **zero dead
  units** — and, crucially, in settings where the EM site is *not* competing with a
  task head at the output: **layer-local objectives** (P004) and the **attention
  transport story** (A002). Frame the trade-off as "not an accuracy trade at the
  output," not as "EM loses."

### 7.6 The λ-Resolution Curve (central figure — promoted from optional)

The "graded" adjective is one of the paper's three load-bearing claims and cannot
rest on three λ points. Run 6–8 λ values × 3 seeds. **X-axis = measured CE/EM
gradient ratio (not λ)** — this is the correct causal variable and the reason the
figure is convincing; Y-axes = conditioning spread (across the 1000× lr range) and
probe accuracy. Prediction: a smooth monotone transition from CE-dominated
(high spread, ~96% probe) to EM-dominated (low spread, ~88% probe), with
stop-gradient sitting at the EM-dominated end. Cost: ~hours on the local GPU,
trivial code (λ is a flag). This is Figure (central).

### 7.7 Basin Analysis / Linear Mode Connectivity (promoted from optional)

Formalizes the fine-tuning claim (contribution 5) and the unpublished "different
basins" observation. Checkpoints exist in `results/experiment4`. Interpolate
parameters between the EM-arm and CE-arm solutions; plot both losses along the
path. **Permutation-align hidden units (Hungarian matching) first** — raw L2 will
not survive review; the barrier plot will. Prediction: a high loss barrier on the
task loss between the two solutions even after alignment → they occupy distinct
basins → discriminative fine-tuning overwrites rather than refines EM structure.
Cost: ~1 day incl. alignment code.

### 7.8 Adam Arm Under Stop-Gradient (completes the Paper 2 signature)

Exp 4 swept SGD only. Run Adam × 4 lrs × 3 seeds in the stop-gradient and joint
λ=0.001 arms. Prediction: Adam ≈ SGD under stop-gradient (no advantage — the third
and final Paper 2 signature), Adam ≫ SGD in the joint arm. Without this, "Paper 2's
signatures reappear" covers only two of three. Cost: ~1 hour, flag exists.

### 7.9 Stochastic-Map Arm (decision: include — it is the bridge to the attention paper)

Replace `W2` with a non-negative, row-normalized (row-stochastic) map. This is the
one preservation class in the taxonomy with no experiment, and — critically — it is
the mechanistic bridge to the companion attention paper (§8.6): attention's value
path *is* a row-stochastic map over the simplex. Prediction: **partial conditioning
survives** (conditioning spread lands between the dense-`W2` and stop-gradient
arms). Risk: the row-normalization constraint may hurt head accuracy enough to
confound the probe — report either way; a null result still populates the taxonomy
cell. A positive result predicts that InfoMax-style volume control transfers to
attention (at-site) while conditioning does not (structural) — exactly the split
the attention paper needs. Cost: ~1 day. **Recommended IN**, given the two papers
are being written as a pair (see ordering note at end).

**Pre-decided fallback (avoid the bridge becoming the softest target).** If the
result comes back confounded by the row-normalization accuracy hit, it moves to the
**appendix** as "taxonomy cell populated, caveated," and does **not** appear in the
abstract or the headline claims. The clean results (Exp 1-4 + λ-curve + basin) carry
the paper; the stochastic-map arm strengthens the taxonomy but is never load-bearing.

## 8. Discussion

### 8.1 Local, Graded, Structural

Return to the main claim.

Local:

- EM lives at LSE/softmax sites.
- It does not automatically propagate through dense maps.

Graded:

- lambda = 0.03 lands between CE-dominated and EM-dominated regimes.
- The relevant x-axis is measured CE/EM gradient ratio.

Structural:

- volume control fixes at-site failures
- conditioning requires private parameters or simplex-compatible transport
- dense learned composition breaks the invariant

### 8.2 What Transfers and What Does Not

Transfers:

- collapse prevention
- dead-unit prevention
- redundancy control
- local mixture calibration

Does not transfer through ordinary dense backprop:

- learning-rate insensitivity
- no Adam advantage
- fixed convergence time
- parameter-independent curvature

### 8.3 Relation to Generative vs Discriminative Training

Likely objection: this is just generative vs discriminative.

Response:

The paper gives the mechanism inside one architecture:

- curvature dichotomy
- orthogonal gradients
- measured dominance
- causal restoration by stop-gradient
- graded restoration by lambda

Ng and Jordan can be cited as the high-level phenomenon; this paper explains
the mechanism through the implicit-EM lens.

### 8.4 Relation to Pretraining and Fine-Tuning

Interpret discriminative fine-tuning as a dominant foreign gradient that can
overwrite local EM structure.

Evidence in hand:

- CE and EM gradients are near orthogonal.
- EM-dominated and CE-dominated solutions occupy different performance regimes.
- Unpublished basin observation suggests large parameter movement between
  objectives.

If basin analysis is added, discuss the barrier plot here.

### 8.5 Design Implications

If we want EM structure at depth:

- create local EM sites
- protect them with stop-gradients or dominance-balanced objectives
- use simplex-compatible transport where possible
- do not expect dense learned maps to preserve the property

This motivates the future layer-wise EM paper.

Attention connection:

Attention's value path is a stochastic map, so it may partially transport
responsibility structure. With §7.9 run, this is a result, not a hope.

### 8.6 Relation to the Attention Project (companion paper)

The companion paper *Depth Is Not Temperature*
(E:\Projects\attention_collapse\paper_proposals) falsifies a depth-as-temperature
theory of transformers and attributes the cause of death to "unpublished
where-EM-lives results": *EM is strictly local to exponentiation+normalization
sites that own an objective, with zero penetration through learned linear maps.*
**That is precisely the locality principle this paper proves and measures in a
controlled setting.** State the relationship explicitly:

- This paper is the *controlled proof* of the locality claim; the attention paper
  is its *observation in the wild* on GPT-2 (a standard transformer owns one site —
  the output CE; intermediate softmax sites are unowned machinery shaped by foreign
  gradient).
- **The depth-as-temperature corollary specifically.** During development the
  program conjectured that stacked LSE sites make depth act as an annealing
  temperature (§1.2, Temptation 2). Locality predicts this cannot hold; *Depth Is
  Not Temperature* tests the corollary on transformers and falsifies it via three
  pre-registered kill criteria. This is the sharpest confirmation of locality
  available: a named, falsifiable depth-schedule prediction that the theory forbids
  and experiment then rejects. Frame atemporally — "locality implies no depth
  schedule; this was tested and confirmed in [A001]" — not "we predicted then
  found," to avoid any post-hoc-storytelling read (the falsification pre-dates the
  clean statement of the locality proof).
- The stochastic-map result (§7.9) is the shared mechanistic hinge: attention's
  value path is row-stochastic, so volume-control/at-site structure can transport
  while conditioning does not. This paper supplies the controlled evidence; the
  attention paper inherits it rather than re-arguing it.
- Dependency/ordering note: P001 (this paper) is the theory; A001 (*Depth Is Not
  Temperature*) is its empirical follow-up. P001 should be citable (arXiv) before
  A001 leans on "confirmation of the locality principle." A001 may be *drafted* in
  parallel (it is largely a writing task) but *posted* after P001.
- One-to-two paragraphs only in the main text; the transformer experiments belong
  to the companion project. This keeps LLMs/attention out of scope here (as the
  proposal requires) while claiming the connection.

## 9. Limitations and Threats to Validity

Split scope-limitations (external validity) from internal-validity threats. A
reviewer treats these differently.

### 9.1 Threats to Internal Validity (address, don't just list)

- **Single feature-quality measure.** The 88%-vs-96% trade-off rests on one
  fixed-protocol linear probe. Robustness check: kNN and a small-MLP probe on the
  same frozen distances; report whether the ordering survives. If it does, the
  trade-off is a property of the representation, not the probe.
- **Curvature measurement.** Spectral norms via power iteration on a fixed
  512-sample batch. Report sensitivity to batch identity/size and iteration count;
  confirm the `1/2` bound and the saturation values are stable.
- **½-saturation diagnostic generality.** Claimed (contribution 4) as usable
  "beyond this paper," but only demonstrated at one site. The depth sweep (§ future
  work / Paper 3) is what earns that generality; state it as promising, not proven,
  until measured at multiple sites.
- **Stop-gradient as the causal knob.** Cutting CE→W1 changes exactly one thing
  (the corridor gradient). State the argument that data, labels, CE, and
  architecture are all held fixed, so the restored conditioning is attributable to
  the corridor and nothing else.

### 9.2 Limitations of Scope (external validity)

- MNIST and small networks.
- Single main architecture.
- No SOTA claims.
- The formal theorem gives sufficient conditions, not a universal iff.
- LayerNorm is included empirically but omitted/simplified in the clean lemma
  unless the proof is extended.
- Experiment 4 currently has SGD-only stop-gradient conditioning; Adam arm is
  still needed for the full Paper 2 optimizer signature.
- Lambda ladder currently has three points; a denser measured-ratio curve would
  strengthen the graded claim.
- No full convergence proof for the composed system.

## 10. Conclusion

Implicit EM is not a property of whole networks. It is a local structure created
by LSE/softmax objectives over distances or energies. When responsibility
gradients reach private parameters directly, the resulting landscape has
parameter-independent curvature and the optimization signatures observed in
the single-layer model. When those gradients pass through ordinary learned dense
maps, the simplex invariant is destroyed and the curvature becomes learned,
anisotropic, and optimizer-sensitive.

The supervised study draws the boundary from both sides. Volume control repairs
local collapse and redundancy at intermediate EM sites. But conditioning
survives only under isolation, dominance, or simplex-compatible structure.
Knowing which properties transfer and which do not is the design rule the
implicit-EM framework now provides.

## Required Work Before Submission

Ranked; all feasible on the local GPU (RTX 3080 Ti, 12 GB + shared).

**Needed for submission:**
1. Adam arms for stop-gradient and joint λ=0.001 (§7.8) — completes the third
   Paper 2 signature. ~1 hour, flag exists.
2. λ-resolution curve on measured CE/EM ratio (§7.6) — the central "graded" figure.
   ~hours.
3. Tighten formal statements: Propositions 1/1'/2 with the boxed scope note
   (§3.1–3.3); Softplus proof, ReLU a.e. + absorbing-state caveat, LayerNorm
   decision. No new math, careful writing.
4. Basin / LMC analysis with Hungarian alignment (§7.7) — formalizes contribution
   5. Checkpoints exist. ~1 day.
5. Related Work section drafted (§2) with the citation set gathered.

**Strengtheners (decide in/out — see ordering note):**
6. Stochastic-map arm (§7.9) — **recommended IN**; bridge to the attention paper.
7. Robustness: Fashion-MNIST, more seeds on headline results, one wider K (§9.2).
8. Threats-to-validity robustness runs (§9.1): alt probes, curvature sensitivity.

**Deferred to Paper 3 of this study (empirical follow-up) if that ordering holds:**
9. Depth sweep (2–6 stacked EM sites, per-layer ½-saturation) — earns the
   diagnostic-generality claim (contribution 4). See ordering note.

## Candidate Figure Plan

1. Conceptual schematic: EM site vs dense corridor (responsibilities live at the
   LSE/softmax site; the dense `W2` pullback destroys simplex structure).
2. Theory panel: private-LSE curvature constant `½·E‖x‖²` vs `W2`-scaled composed
   curvature (Prop 1 vs Prop 2).
3. Exp 1 ablation table/plot (at-site volume-control repair).
4. Exp 2 capacity plot (dead units eliminated across widths; redundancy slope).
5. Exp 3 optimizer sweep (Paper 2 signature disappears under composition).
6. Exp 4 conditioning sweep (probe accuracy vs lr, four arms).
7. Exp 4 gradient competition (CE/aux ratio + cosine).
8. Exp 4 curvature and ½-saturation (LSE ≤ ½; pure EM saturates; CE-path larger).
9. **λ-resolution curve** (central; x = measured CE/EM ratio) — promoted to main.
10. **Basin interpolation plot** (aligned; task-loss barrier) — promoted to main.
11. Predictions ledger (§7.0) rendered as a compact table/figure.

## Reviewer Feedback Triage (round 1 — applied)

Read-through feedback, triaged. Applied items are folded into the sections above.

**Required (applied):**
1. *Punchline vs λ=1 tension* — the "parameter sharing in disguise" sentence and
   the "dominance not connectivity" result can be quoted against each other.
   Fixed: split structure (why the foreign term is ill-conditioned, Prop 3) from
   dominance (when it wins). Reconciliation paragraph added to §1.3; guardrail
   added to §3.5. **This was the one to fix before drafting.**
2. *Prop 2 oversells* — it is an upper bound (guarantee forfeited), not proof of
   large curvature. Renamed "Composition Forfeits the Uniform Bound"; abstract
   softened; §7.4 carries the "actually large" weight.
3. *88/96 deflationary reading* — pre-empted in §7.5 (continuum via λ-curve;
   "who wants conditioning" when the cost is paid off-output).
4. *Related-work gaps* — added gradient-conflict/multi-task (PCGrad, GradNorm,
   gradient starvation) and local-learning (Belilovsky, synthetic gradients,
   Forward-Forward) to §2.

**Local surgery (applied):**
5. Cosine ≈ 0 → report vs chance band (±1/√K ≈ 0.2), §7.3.
6. Gradient-competition measured under Adam, sweep under SGD → add SGD measurement
   or argue optimizer-independence, §7.3.
7. Convergence-time metric odd in CE-dominated arm (nobody optimizes LSE there) →
   reframed in §7.2.
8. Prop 1 constant ½·E‖x‖² is a checkable lr prediction (≈2/L) → upgrade to
   quantitative, §3.2.
9. §2b/§2 numbering collision → §2.5 with a promote-in-final note.

**Scope discipline (applied):**
10. Stochastic-map arm: pre-decided appendix fallback if row-norm confounds the
    probe; never load-bearing, §7.9. Depth sweep stays deferred to P004.

**Pre-submission checklist (NOT yet done — do before anything leaves the machine):**
- **Scrub all `E:\...` absolute paths** (currently in §2, §8.6, and brainstorm.md)
  → replace with repo-relative references or citations.
- Confirm arXiv IDs and author-year for every citation added in §2.
- Final section renumber (Related Work → §2, Background → §3, cascade).

