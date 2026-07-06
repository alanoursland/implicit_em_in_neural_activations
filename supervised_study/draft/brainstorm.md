# Brainstorm: Paper From the Supervised Study

This is a collection file, not an outline. It gathers the claims, evidence,
numbers, open tasks, and framing options for the paper currently described in
`paper_proposals/failure_conditions.md`.

Working old name: "The Conditions Under Which Implicit EM Fails".

Probably wants a different name. Possible directions:

- Where Implicit EM Lives
- The Boundary of Implicit EM
- Locality of Implicit EM in Neural Networks
- When Responsibility-Weighted Gradients Survive Backpropagation
- Why Backpropagation Breaks Implicit EM
- Local, Graded, Structural: Failure Modes of Implicit EM
- The Structural Limits of Implicit EM in Neural Networks
- Volume Control Transfers, Conditioning Does Not
- Backpropagation as Parameter Sharing: A Failure Mode of Implicit EM

## High-Level Paper Shape

Paper 1 derived the gradient-responsibility identity:

```text
For log-sum-exp objectives over distances/energies,
dL/dd_j = -r_j
```

Responsibilities arise as gradients. Gradient descent on these objectives
therefore performs generalized EM implicitly.

Paper 2 showed the theory is prescriptive in the unsupervised regime: build a
decoder-free encoder from LSE + volume control, and it behaves like the theory
predicts. It also found the unusual optimization signature:

- SGD was insensitive to learning rate over a wide range.
- Adam offered little/no advantage.
- Convergence time was roughly fixed.
- Lower loss did not necessarily mean better features.

This supervised-study paper should answer the natural next question:

When does that EM optimization property actually hold inside a neural network,
and what destroys it?

The current best claim:

Implicit EM is local, graded, and structural.

- Local: the EM property exists at exponentiation + normalization sites and
  extends zero learned dense layers beyond them.
- Graded: in a real network, behavior varies continuously with the share of the
  arriving gradient that is EM-structured. The transition is controlled by the
  measured CE/EM gradient dominance ratio, not by connectivity alone.
- Structural: failures split into at-site failures and structural failures.
  At-site failures can be repaired locally. Structural failures cannot be
  repaired by volume control because the decomposition EM needs no longer
  exists.

One-line punchline:

Backpropagation composition is parameter sharing in disguise. Classical EM loses
its clean M-step when components share parameters; neural implicit EM loses its
conditioning when responsibility gradients are pulled through shared learned
maps.

## Central Distinction

At-site failures:

- collapse
- dead components
- redundant components
- uninformative responsibilities
- hard assignment / ReLU death

Remedies:

- variance penalty
- decorrelation penalty
- softer kernels
- temperature
- volume control applied at the site

These remedies transfer to supervised intermediate layers.

Structural failures:

- dense sign-indefinite learned maps
- objective mixing
- shared lower parameters across components
- non-exponential-family/deep component maps

Remedies:

- stop-gradient / isolation
- make the EM objective dominant
- create EM sites locally at each layer
- possibly transport through simplex-compatible maps, e.g. stochastic maps

Regularization cannot restore the original EM decomposition once parameters are
shared across components.

Useful slogan:

Volume control transfers anywhere; conditioning transfers nowhere unless the
structural conditions are preserved.

## What This Paper Should Not Claim Too Strongly

Avoid an unrestricted "if and only if".

What seems defensible:

- The paper proves sufficient conditions for the conditioning property:
  private parameters, LSE structure, simplex-compatible pullbacks, monotone
  per-coordinate kernels.
- It demonstrates failures under violated conditions:
  dense learned map, objective mixing, hard gates.
- It shows empirically that dominance controls the transition in the studied
  network.

What is not proved:

- Necessity of the conditions in all possible neural architectures.
- A general theory of deep-network conditioning.
- Convergence proofs for the composed supervised system.

## Theory Ingredients

### Where EM Lives

Source: `notes/where_em_lives.md`.

The identity applies at the interface where the loss sees distances/energies.
For cross-entropy:

```text
L = -z_y + log sum_k exp(z_k)
dL/dz_j = softmax(z)_j - 1[j = y]
```

The output layer receives label-clamped responsibility gradients.

But internal layers do not automatically become EM layers. The responsibility
vector is transformed by intermediate Jacobians. By the time it reaches an
earlier dense layer, it is no longer a posterior over that layer's units.

Correct claim:

The loss geometry imposes EM structure at the output, and gradients from this
structure flow backward to shape internal representations.

Incorrect claim:

The whole network is doing EM.

Implication:

If we want EM structure inside the network, it must be created locally:

- auxiliary LSE objectives at layers
- competitive normalizations inside layers
- layer-wise local objectives
- simplex-compatible transport

### Conditioning Lemma

Source: `notes/conditioning_lemma.md`.

For LSE over distances:

```text
L(d) = -log sum_j exp(-d_j)
r = softmax(-d)
grad_d L = r
H_d L = rr^T - diag(r)
||H_d L||_2 = lambda_max(diag(r) - rr^T) <= 1/2
```

The bound is uniform in d. No parameters appear.

For private linear energies:

```text
d_j = w_j^T x + b_j
||grad_theta L|| <= ||x||
||H_theta L|| <= (1/2)||x||^2
```

For batch-mean loss:

```text
||H|| <= (1/2) E ||x||^2
```

This is a data constant, invariant over training. It explains the Paper 2
optimization signature: the admissible learning-rate range does not shrink as
parameters move, and there is no learned anisotropy for Adam to exploit.

For smooth monotone kernels such as Softplus, the chain rule adds bounded
diagonal factors and bounded second-order terms. Parameter-independence survives
up to constants. ReLU satisfies bounds almost everywhere but has an absorbing
state pathology.

For composition through a learned map:

```text
h = W2 y
y = NLS(d) = d + log sum_k exp(-d_k)
CE(h, c) = -log softmax(h)_c
```

The Hessian w.r.t. d contains:

```text
J^T W2^T (diag(p) - pp^T) W2 J
```

plus terms from the curvature of NLS. The key scaling is:

```text
||H_d CE|| = O(sigma_max(W2)^2)
```

The uniform 1/2 bound is replaced by a learned, changing, anisotropic quantity.

This is the clean theorem/lemma pair for the paper:

- Private-parameter LSE: bounded by a data constant.
- Same EM site behind a learned dense map: curvature scales with W2.

### Operations That Preserve or Destroy the Invariant

Source: `notes/why_composition_breaks_em.md`.

Invariant:

The gradient is a conditional expectation on the simplex:

- nonnegative
- normalized
- per-component aligned
- bounded
- Hessian bounded by the Bohning/Popoviciu 1/2 bound

Preserving operations:

- private per-component parameters
- per-coordinate monotone maps
- block-diagonal/private pullbacks
- stochastic/Markov maps, at least first order

Destroying operations:

- dense sign-indefinite learned linear maps
- objective mixing
- hard gates
- shared parameters across components

Stochastic maps are important as an open bridge to attention. A row-stochastic
matrix maps the simplex to itself; its transpose preserves non-negativity and
total gradient mass under pullback. This is why attention may be special.

### Classical EM Connection

Structural failure in this paper is the neural version of known EM failures.

Table to include or adapt:

| Classical EM failure | Why EM fails | Neural incarnation |
|---|---|---|
| Shared parameters across components | Q no longer separates; M-step coupled | Backprop composition shares lower layers across all hypotheses |
| Non-exponential-family complete-data model | No closed-form M-step; nonconvex inner problem | Component energies computed by deep networks |
| Unbounded likelihood / singular components | Component collapse; variance goes to zero | LSE-only collapse; needs volume control |
| Hard assignment | Zero-responsibility components cannot recover | Dead ReLU units |
| High missing information | Slow convergence with uniform responsibilities | Diffuse responsibilities; temperature issue |

Main classical-EM punchline:

EM's efficiency comes from the E-step making the objective separate over
components, so the M-step decouples into independent per-component problems.
That requires each component to own its parameters. Backpropagation composition
violates that condition by making lower parameters shared by all upper
hypotheses.

## Model / Experimental Setup

The supervised study uses a small MNIST classifier. Main architecture across
reports:

```text
x in R^784
-> Linear(784, K)
-> ReLU
-> distances d
-> optional NegLogSoftmin / volume-control site
-> Linear(K, 10)
-> LayerNorm
-> CE
```

Experiment 4 version:

```text
x -> Linear(784, 25) -> ReLU -> d
  -> NegLogSoftmin
  -> Linear(25, 10) -> LayerNorm -> CE

auxiliary loss on d: LSE + var + tc
```

NegLogSoftmin:

```text
y_j = d_j + log Z
Z = sum_k exp(-d_k)
exp(-y_j) = responsibility_j
```

It is a type-preserving calibration: distances in, calibrated distances out.
It embeds the LSE partition function and introduces a competitive Jacobian.

Volume control:

- LSE term creates responsibility structure.
- Variance term prevents dead/collapsed units.
- Decorrelation/total-correlation term prevents redundant components.
- This is the neural analogue of the log-determinant in mixture models.

## Experiment 1: Volume Control Ablation

Source: `supervised_study/reports/1_ablation_report.md`.

Question:

Can EM be extended from the output layer into an intermediate layer of a
supervised network, and does the same volume-control requirement from Paper 2
apply?

Design:

- MNIST
- hidden dim 25
- 50 epochs
- Adam lr = 0.001
- lambda_reg = 0.001
- 10 seeds, 42-51
- summary over epochs 41-50

Configs:

| Config | NLS | Var | Decorr | Role |
|---|---:|---:|---:|---|
| Baseline | no | no | no | ordinary MLP |
| NLS only | yes | no | no | EM without volume control |
| NLS + Var | yes | yes | no | anti-collapse only |
| NLS + Var + Decorr | yes | yes | yes | full volume control |
| Var + Decorr only | no | yes | yes | regularization without EM |

Main results:

| Config | Dead Units | Min Var | Redundancy | Resp Entropy | Accuracy |
|---|---:|---:|---:|---:|---:|
| Baseline | 1.40 +/- 0.49 | 0.000 +/- 0.000 | 20.5 +/- 2.8 | 3.08 +/- 0.02 | 96.09% +/- 0.07% |
| NLS only | 1.20 +/- 0.98 | 0.264 +/- 0.408 | 24.9 +/- 3.0 | 2.93 +/- 0.02 | 96.14% +/- 0.17% |
| NLS + Var | 0.00 +/- 0.00 | 11.74 +/- 3.08 | 187.0 +/- 38.7 | 2.09 +/- 0.09 | 95.75% +/- 0.21% |
| NLS + Var + Decorr | 0.00 +/- 0.00 | 7.84 +/- 1.17 | 13.9 +/- 0.5 | 2.46 +/- 0.04 | 96.34% +/- 0.11% |
| Var + Decorr only | 0.10 +/- 0.30 | 3.55 +/- 1.49 | 13.3 +/- 0.9 | 2.65 +/- 0.07 | 96.40% +/- 0.10% |

Findings:

- Supervised gradients do not provide volume control at intermediate layers.
  Baseline has min variance 0 across all seeds; at least one dead ReLU in every
  trained model.
- NLS alone is statistically indistinguishable from baseline. EM without volume
  control does not organize representation.
- Partial volume control is worse than none: NLS + Var eliminates dead units
  but redundancy explodes to 187 and accuracy is worst.
- Full volume control stabilizes the representation: zero dead units, low
  redundancy, best/tied accuracy.
- Var + Decorr without NLS is good regularization but not the same as an
  EM-structured mixture. It has higher responsibility entropy and lower min var.
- Baseline achieves low CE loss despite worse representation; geometry and CE
  objective are not the same thing.

Use in final paper:

This is the at-site half. Collapse/deadness/redundancy live at the EM site and
are repaired by volume control applied there, even inside a supervised network.

## Experiment 2: Capacity and Volume Control

Source: `supervised_study/reports/2_capacity_volume_report.md`.

Question:

Does the volume-control effect persist across hidden width?

Design:

- Compare baseline vs NLS + Var + Decorr.
- hidden dims: 16, 25, 36, 49, 64
- MNIST
- 50 epochs
- Adam lr = 0.001
- lambda_reg = 0.001
- 5 seeds, 42-46

Main results:

| Hidden | Baseline Dead | VC Dead | Baseline Redundancy | VC Redundancy | Baseline Acc | VC Acc |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1.0 +/- 0.9 | 0.0 | 11.0 +/- 2.2 | 9.1 +/- 0.9 | 95.19% | 95.32% |
| 25 | 1.4 +/- 0.5 | 0.0 | 19.0 +/- 1.3 | 13.8 +/- 0.5 | 96.07% | 96.35% |
| 36 | 2.4 +/- 1.9 | 0.0 | 31.7 +/- 5.1 | 21.1 +/- 1.5 | 96.63% | 96.60% |
| 49 | 2.5 +/- 0.9 | 0.0 | 48.1 +/- 3.6 | 29.7 +/- 0.5 | 97.04% | 96.87% |
| 64 | 4.1 +/- 1.3 | 0.0 | 69.0 +/- 7.2 | 41.2 +/- 0.8 | 97.29% | 97.00% |

Findings:

- Dead units appear at every capacity in the baseline, with stable dead fraction
  around 5-7%.
- Volume control eliminates dead units at every capacity.
- Redundancy scales roughly linearly with width in the baseline. Volume control
  reduces redundancy at every width and tightens seed variance.
- Accuracy benefit depends on capacity and lambda. It helps at 16 and 25, is
  neutral at 36, slightly hurts at 49 and 64 under fixed lambda = 0.001.
- Separate calibration suggests 64 prefers smaller lambda = 0.0001, where VC
  can still produce a small accuracy gain.
- Structural benefits are robust; accuracy benefits require calibration.

Interpretation:

- Capacity-constrained regime: every unit matters; volume control helps both
  structure and accuracy.
- Capacity-abundant regime: extra units can act as correlated boundary refiners;
  decorrelation may penalize useful cooperation.
- Overparameterization and volume control solve related problems differently:
  overparameterization tolerates dead/redundant units; volume control makes each
  unit alive and distinct.

Use in final paper:

This supports volume control as an at-site repair across capacity and prevents
overstating accuracy claims. The paper should not be about SOTA or accuracy.

## Experiment 3: Optimization Dynamics

Source: `supervised_study/reports/3_optimizer_report.md`.

Question:

Do Paper 2's optimization signatures survive when an EM-structured layer is
inside a supervised network?

Design:

- Config: NLS + Var + Decorr
- optimizers: SGD and Adam
- learning rates: 0.0001, 0.001, 0.01, 0.1
- MNIST
- hidden dim 25
- 50 epochs
- lambda_reg = 0.001
- 3 seeds, 42-44

Main results:

| Optimizer | LR | Accuracy | CE Loss | Reg Loss | Min Var | Redundancy |
|---|---:|---:|---:|---:|---:|---:|
| SGD | 0.0001 | 83.83% +/- 4.96% | 0.868 | 107.5 | 0.015 | 27.5 |
| SGD | 0.001 | 93.69% +/- 0.26% | 0.275 | 69.9 | 0.065 | 19.6 |
| SGD | 0.01 | 95.63% +/- 0.12% | 0.155 | 19.2 | 0.436 | 17.7 |
| SGD | 0.1 | 96.23% +/- 0.09% | 0.133 | -34.3 | 4.01 | 16.2 |
| Adam | 0.0001 | 95.34% +/- 0.33% | 0.170 | 24.6 | 0.324 | 18.0 |
| Adam | 0.001 | 96.38% +/- 0.11% | 0.133 | -57.3 | 7.80 | 13.8 |
| Adam | 0.01 | 96.37% +/- 0.12% | 0.160 | -158.3 | 423.2 | 11.9 |
| Adam | 0.1 | 96.58% +/- 0.05% | 0.179 | -265.9 | 30628.4 | 11.9 |

Findings:

- Paper 2's SGD learning-rate insensitivity does not replicate in the ordinary
  supervised joint setting. SGD accuracy spans 83.8% to 96.2%.
- Adam clearly outperforms SGD. Adam at lr 0.0001 nearly matches SGD at 0.01;
  Adam at 0.001 matches/exceeds SGD at 0.1.
- High-lr Adam shows a partial Paper 2 loss-feature decoupling: reg loss and
  min variance explode without meaningful accuracy gains.
- Adam lr = 0.001 is the practical operating point.

Original interpretation, corrected by experiment 4:

- The CE path through W2 scrambles the responsibility structure.
- Objective mixing makes the EM signal a small perturbation on a dominant
  supervised signal.
- ReLU masking creates absorbing dead states.
- Report 3 inferred a roughly 1000:1 dominance from lambda; experiment 4 later
  measured 30-70x instead. Keep the mechanism, correct the magnitude.

Use in final paper:

This is the first observation that the optimization property does not survive
composition. It motivates experiment 4.

## Experiment 4: Why Composition Breaks EM Conditioning

Source: `supervised_study/reports/4_composition_report.md`.

Question:

Is composition really the cause, and what mechanism destroys conditioning?

Predictions:

- P1: Gradient competition. At small lambda, CE gradient reaching the
  intermediate distances dominates the auxiliary EM gradient. The two are
  near-orthogonal.
- P2: Curvature dichotomy. LSE Hessian w.r.t. d stays bounded by 1/2. CE-path
  Hessian w.r.t. d is parameter-dependent and scales with W2.
- P3: Causal test. Stop the CE gradient from reaching W1, and Paper 2's
  conditioning signatures should reappear inside the supervised network.

Design:

Model:

```text
x -> Linear(784, 25) -> ReLU -> d -> NLS -> Linear(25, 10) -> LayerNorm -> CE
aux(d) = LSE + var + tc
```

Arms:

| Arm | Total loss | CE reaches W1? | Regime |
|---|---|---:|---|
| joint, lambda = 0.001 | CE + 0.001 aux | yes | CE-dominated |
| joint, lambda = 0.03 | CE + 0.03 aux | yes | balanced |
| joint, lambda = 1 | CE + 1.0 aux | yes | EM-dominated |
| stop-gradient | CE(head(y.detach())) + 1.0 aux | no | pure EM at W1 |

Conditioning sweep:

- arms x SGD lr in {0.0001, 0.001, 0.01, 0.1}
- 3 seeds, 42-44
- 40 epochs
- feature quality measured by fixed-protocol linear probe on frozen distances

Gradient competition:

- during joint-arm Adam training at lr = 0.001
- every 100 steps compute ||grad_d CE|| and lambda ||grad_d aux||
- compute cosine similarity

Curvature:

- every 2 epochs
- power iteration with exact HVPs
- spectral norm of Hessian w.r.t. d for LSE and CE path
- fixed 512-sample batch
- record sigma_max(W2)

P3 results: linear probe accuracy on frozen distances:

| Arm | lr=0.0001 | lr=0.001 | lr=0.01 | lr=0.1 | Spread |
|---|---:|---:|---:|---:|---:|
| joint, lambda=0.001 | 85.32% +/- 3.13% | 92.93% +/- 0.22% | 95.09% +/- 0.14% | 96.21% +/- 0.18% | 10.9 pts |
| joint, lambda=0.03 | 86.37% +/- 0.29% | 90.57% +/- 0.11% | 91.83% +/- 0.16% | 92.24% +/- 0.38% | 5.9 pts |
| joint, lambda=1 | 88.39% +/- 0.05% | 88.35% +/- 0.08% | 88.59% +/- 0.35% | 87.41% +/- 0.23% | 1.2 pts |
| stop-gradient | 88.35% +/- 0.06% | 88.29% +/- 0.17% | 88.18% +/- 0.25% | 87.03% +/- 0.30% | 1.3 pts |

Interpretation:

- Stop-gradient restores learning-rate insensitivity across a 1000x range.
- joint lambda = 1 is indistinguishable from stop-gradient.
- lambda = 0.03 lands between regimes.
- Conditioning follows dominance, not connectivity.

Convergence-time result:

Epoch at which test LSE reaches 95% of final value:

| Arm | lr=0.0001 | lr=0.001 | lr=0.01 | lr=0.1 |
|---|---:|---:|---:|---:|
| stop-gradient | 32.0 | 27.7 | 21.7 | 23.7 |
| joint, lambda=0.001 | 32.3 | 15.0 | 40.0/never | 37.7 |

Stop-gradient is much more stable across lr and seeds; joint is erratic.

P1 results: gradient competition:

| Arm | CE / weighted aux early | CE / weighted aux late | Cosine |
|---|---:|---:|---:|
| joint, lambda=0.001 | 30-35x | 62-71x | approx 0 |
| joint, lambda=1 | 0.042-0.046x | 0.021x | approx 0 |

Important correction:

The CE/EM ratio at lambda = 0.001 is 30-70x, not 1000x. Lambda itself does
not determine the gradient ratio because the raw aux gradient is larger.

Important interpretation:

The CE and EM gradients are orthogonal, not opposed. The EM signal is drowned,
not fought.

P2 results: curvature:

- LSE Hessian stays below 1/2 at every measurement in every arm and seed.
- In stop-gradient / pure EM, LSE curvature saturates the bound exactly:
  0.4993-0.4997.
- This is achieved when responsibility mass concentrates equally on two
  components at a boundary.
- Therefore distance from 1/2 is a diagnostic of mixture sharpness, not just a
  worst-case bound.
- In CE-dominated training, LSE curvature stagnates around 0.15-0.26; the
  mixture structure stays diffuse.
- CE-path curvature in representative joint lambda=0.001 trajectory wanders
  around 6.5 -> 17.2 -> 7.8, roughly 13-100x larger than LSE curvature,
  non-monotone, consistent with W2 scaling.

Core findings:

- Composition, not supervision, destroys conditioning.
- Stop-gradient arm is still supervised: same data, labels, CE, architecture.
  Only the CE-to-W1 path is removed.
- Conditioning is graded by measured gradient dominance.
- Bohning bound saturation gives a new diagnostic.
- Restored conditioning has a cost: EM-dominated features probe at about 88%,
  while CE-dominated features probe at about 96%.

Honest trade-off:

The EM regime buys optimization properties: learning-rate robustness,
reproducibility, convergence-time invariance, zero dead units. It does not buy
task accuracy. Well-conditioned optimization of the wrong/less-aligned objective
does not beat ill-conditioned optimization of the task objective.

Use in final paper:

This is the core causal evidence and likely the centerpiece.

## Figures / Tables To Consider

Possible central figures:

- Conceptual diagram: EM site vs corridor. Responsibilities live at LSE/softmax
  site; dense W2 pullback destroys simplex structure.
- Theory figure: private-parameter LSE has curvature bound 1/2 E||x||^2;
  composed CE path has W2^T H W2 scaling.
- Experiment 1 ablation table: at-site volume control repairs dead/redundant
  components.
- Experiment 2 capacity curve: dead units eliminated across widths; redundancy
  reduced; accuracy trade-off by capacity.
- Experiment 3 optimizer table: Paper 2 optimization signature disappears under
  joint supervised composition.
- Experiment 4 conditioning sweep: probe accuracy vs lr for four arms.
- Experiment 4 gradient competition: CE/aux ratio and cosine.
- Experiment 4 curvature: LSE <= 1/2, CE path much larger, pure EM saturates
  1/2.
- Future/optional lambda-resolution curve: x-axis measured CE/EM gradient ratio,
  y-axis conditioning spread and probe accuracy.
- Optional basin plot: linear mode connectivity between EM-dominated and
  CE-dominated checkpoints.

## Possible Contributions

1. Characterize when the implicit-EM optimization property holds: private
   parameters, LSE/softmax sites, simplex-compatible operations, monotone local
   kernels.
2. Show why it fails under neural composition: dense learned maps convert a
   universal curvature bound into W2-dependent curvature.
3. Introduce the at-site vs structural failure taxonomy.
4. Demonstrate at-site transfer: volume control fixes dead/redundant intermediate
   units in supervised networks.
5. Demonstrate structural failure: conditioning does not survive composition.
6. Establish causally that the corridor is the cause: stop-gradient restores
   Paper 2 signatures inside a supervised network.
7. Show the transition is graded by measured gradient dominance.
8. Introduce 1/2-saturation as a diagnostic of mixture sharpness at an EM site.
9. Explain why generative/EM-like pretraining can be overwritten by
   discriminative fine-tuning: dominant orthogonal gradients move into a
   different basin; nothing preserves the local EM structure.

## Possible Abstract Ingredients

Potential abstract skeleton:

Paper 1 showed that log-sum-exp objectives produce responsibility gradients,
so gradient descent performs implicit EM at the objective interface. But this
does not imply that a deep network inherits EM's optimization properties
through backpropagation. We characterize the boundary. The LSE Hessian in
energy coordinates is uniformly bounded by 1/2; with private component
parameters this gives parameter-independent curvature bounded by a data
constant. When the same site is composed behind a learned dense map, the
curvature contains W^T(diag p - pp^T)W and scales with the learned spectrum.
Thus composition turns a locally conditioned EM update into an ordinary
parameter-dependent deep-network gradient. Experiments in supervised MNIST
networks confirm the split. Volume control applied at an intermediate EM site
eliminates dead units and redundancy across widths, showing that at-site
failures transfer. But Paper 2's optimization signatures disappear under
ordinary supervised composition. Measuring gradients shows that the CE path
dominates the EM path by 30-70x and is nearly orthogonal to it; cutting that
path with stop-gradient restores learning-rate insensitivity and fixed
convergence time. Increasing the EM weight produces a continuous transition.
The result is a local, graded, structural account of implicit EM: volume
control transfers through networks, but EM conditioning survives only where
responsibility gradients reach private parameters through simplex-compatible
operations.

## Possible Introduction Flow

1. Start from the strong result:
   LSE gradients are responsibilities; neural objectives instantiate Fisher's
   identity/implicit EM directly.
2. The tempting overclaim:
   If neural objectives do implicit EM, perhaps deep networks inherit EM's
   conditioning everywhere.
3. Paper 2 made this temptation stronger:
   single-layer decoder-free model trained with surprising optimizer
   insensitivity.
4. But deep networks are composed systems. Backpropagation transforms the
   responsibility vector before it reaches lower parameters.
5. This paper asks where the EM property survives.
6. Answer: it is local, graded, structural.
7. Contributions and evidence.

## Relation To Paper 1 and Paper 2

Paper 1:

- Algebraic identity.
- Training-time responsibility gradients.
- Unified regimes: unsupervised mixture modeling, attention, supervised
  cross-entropy.

Paper 2:

- Prescriptive unsupervised model.
- LSE + InfoMax/volume control.
- Showed failure modes from missing volume control.
- Found optimizer anomaly: SGD insensitive, Adam no advantage, fixed
  convergence time.

This paper:

- Explains when Paper 2's optimizer anomaly should exist.
- Shows it is not a generic property of "having an EM-ish layer".
- Shows volume-control failure modes are at-site and repairable.
- Shows conditioning is structural and fragile under composition.

## Relation To Supervised Networks

Old framing:

"Does implicit EM theory correctly predict the behavior of intermediate layers
in supervised networks?"

New framing:

"Which parts of implicit EM transfer to intermediate layers, and which parts
do not?"

Answer:

- Volume control transfers.
- Optimization conditioning does not, unless the EM gradient dominates or the
  foreign path is cut.

Labels provide volume control at the output layer but not at intermediate
layers. Supervised gradients can train useful features, but they do not prevent
dead/redundant units by the same mechanism.

## Relation To Generative vs Discriminative

Likely reviewer objection:

"Isn't this just generative vs discriminative training?"

Response:

The paper is not merely observing that objectives differ. It identifies the
mechanism:

- curvature dichotomy
- gradient orthogonality
- measured dominance
- causal restoration by stop-gradient
- graded restoration by lambda

Ng & Jordan can be cited for the broader phenomenon. This paper gives the
inside-one-architecture mechanism.

## Relation To Pretraining / Fine-Tuning

Claim to handle carefully:

Generative/EM-style pretraining does not automatically survive discriminative
fine-tuning because the discriminative gradient is a dominant, orthogonal,
structurally different signal.

Evidence in hand:

- Experiment 4: CE and EM gradients at d are near orthogonal.
- EM-dominated and CE-dominated regimes land at different feature-quality
  plateaus.
- Unpublished Oursland observation: EM-pretrained encoder fine-tuned by GD, and
  GD-trained autoencoder fine-tuned by the EM objective, move enormous L2
  distances in parameter space.

Needs stronger analysis for paper:

- Linear mode connectivity between EM-arm and CE-arm solutions.
- Interpolate parameters, plot both losses along path.
- Use Hungarian matching on hidden units to avoid permutation artifacts.

## Open Work / Required Before Submission

From proposal, ranked:

1. Adam arm under stop-gradient:
   Experiment 4 swept SGD only. Need Adam x 4 lrs x 3 seeds in stop-gradient
   and joint lambda=0.001 arms. Prediction: Adam approximately equals SGD under
   stop-gradient; Adam much better than SGD in joint arm.

2. Lambda-resolution curve:
   Current ladder has 0.001, 0.03, 1. Need 6-8 lambda values. X-axis should be
   measured CE/EM gradient ratio, not lambda. Y-axis: conditioning spread and
   probe accuracy.

3. Formal tightening:
   - Lemma currently cleanest for linear energies.
   - Extend to monotone kernels with bounded phi' and phi''.
   - Softplus should be straightforward.
   - ReLU should be stated as a.e. bound plus absorbing-state caveat.
   - Either include LayerNorm constants or define the theoretical model without
     LayerNorm.
   - Replace "iff" language with sufficient conditions + demonstrated failures.

4. Basin analysis:
   Linear mode connectivity between EM-arm and CE-arm solutions. Existing
   checkpoints likely in experiment4 results. Need alignment before
   interpolation.

5. Depth sweep:
   Stack 2-6 EM layers with local LSE + VC sites. Train jointly and with
   per-layer stop-gradients. Plot 1/2-saturation diagnostic per layer.
   This may seed the future layer-wise EM paper.

6. Stochastic-map arm:
   Replace W2 with non-negative row-normalized map. Prediction: partial
   conditioning survives. This is the bridge to attention.

7. Robustness:
   Fashion-MNIST, more seeds on headline results, one wider K.

8. Writing:
   New outline replacing old confirmation-framed outline.

## Out Of Scope

- LLMs and transformer experiments, except as discussion/future connection.
- Attention-sink experiments; those belong to the attention paper.
- SOTA or accuracy claims.
- General theory of all deep-network conditioning.
- Full convergence proof for composed systems.
- Claiming all internal representations are mixtures.

## Positioning / Citations To Gather

Need likely sources:

- Dempster, Laird, Rubin: EM.
- Neal and Hinton: free-energy view of EM.
- Xu and Jordan: EM as preconditioned gradient / convergence behavior.
- Bohning / multinomial logistic Hessian bound.
- Popoviciu inequality for variance bound.
- Ng and Jordan: generative vs discriminative classifiers.
- Linear mode connectivity: Frankle et al. or related.
- Classical EM failures with shared parameters / generalized EM.
- Layer-wise pretraining / DBNs.
- Attention as future bridge; maybe not in main positioning except discussion.

## Risk Register

"This is just regularization."

Answer:

Experiment 1 shows Var + Decorr without NLS is good regularization, but NLS
changes responsibility entropy and calibration. More importantly, Experiment 4
is not about accuracy regularization; it directly measures curvature and
gradient dominance.

"This is just MNIST."

Answer:

The claims are about optimization structure and exact constants, not vision
performance. Still, Fashion-MNIST robustness would help.

"Volume control only marginally helps accuracy."

Answer:

Accuracy is not the main claim. Structural health and conditioning are the
claim. Also, the honest trade-off is central: EM-dominated features are less
task-aligned.

"Why use ReLU if ReLU itself breaks EM?"

Answer:

This is part of the taxonomy. ReLU is a hard gate / absorbing-state failure.
Formal statements should separate smooth monotone kernels from ReLU a.e.
behavior.

"The full model includes LayerNorm but the lemma omits it."

Answer:

Need either include it in constants or state the theoretical model without
LayerNorm and explain the empirical measurements absorb its effect.

"Iff is too strong."

Answer:

Use sufficient conditions plus empirical demonstrations of violations.

## Possible Final Contribution Paragraph

Implicit EM gives neural objectives a locally well-conditioned responsibility
geometry, but only where its structural assumptions hold. For LSE objectives,
the energy-space curvature is uniformly bounded by 1/2; with private component
parameters this becomes a parameter-independent data constant. Learned dense
composition replaces that constant with the spectrum of the intervening map,
and objective mixing decides which landscape dominates. Experiments in a
supervised network show the resulting split: volume control repairs local
collapse and redundancy at intermediate EM sites, but conditioning is lost
through the backpropagation corridor and restored only by isolation or
dominance. The implicit-EM property is therefore not a property of whole
networks. It is a local structure that can be created, destroyed, transported,
or protected.

---

## Session Addenda (decisions made while revising outline.md)

These were resolved during the outline revision and cross-project review. They
supersede the corresponding "open" items above.

### Title — decided

**Recommended: "Volume Control Transfers, Conditioning Does Not: The Locality of
Implicit EM in Neural Networks."** Most information-dense candidate; states the
one-sentence contribution directly. Commit early — it fixes the abstract shape.

### Depth-as-temperature — the second temptation (honest framing)

During development the program conjectured that stacked LSE sites make **depth act
as an annealing temperature schedule**. P001's locality result *forbids* this: a
distributed schedule is exactly an EM property propagating through composition,
which Proposition 2 rules out. The companion paper **A001 (Depth Is Not
Temperature)** tested the corollary on transformers and **falsified it** via three
pre-registered kill criteria.

How to use it in P001 (settled):
- Present depth-as-temperature as **Temptation 2** in §1.2, alongside the
  "whole network inherits EM" temptation. It is Temptation 1 sharpened into a
  falsifiable corollary.
- State it as **a prediction the theory makes, not a hypothesis the paper
  endorses.** P001 never claims temperature is real. It says locality predicts no
  schedule, and points to A001 for the empirical kill.
- **Frame atemporally** — "locality implies no depth schedule; tested and confirmed
  in [A001]" — NOT "we predicted, then found." The falsification pre-dates the
  clean locality statement; avoid any post-hoc-storytelling read.
- This is a strength: a theory that correctly forbids a named falsification is more
  convincing than one that only fits confirmations.

### Cross-project dependency and ordering (confirmed against a_publication_plan)

- **P001 is the theory; A001 is its empirical follow-up.** A001's own proposal
  attributes its cause of death to "unpublished where-EM-lives results" — which is
  exactly P001's locality proof. So the true dependency is **P001 → A001**
  (and → A002, → P004, which use the at-site/structural taxonomy).
- The publication plan (`E:\Projects\a_publication_plan`) currently has an
  inversion: `proposed-order.md` ranks A001 (2) ahead of P001 (3) on *readiness*
  grounds (A001 is nearly pure writing), but writes it as if it were *dependency*.
  The P001 card itself says "provisional order 1" and "gate for" downstream papers,
  contradicting the table. **User will have the plan's manager fix the summary.**
- Resolution rule: **decouple drafting order from publication order.** A001 can be
  *drafted* in parallel (writing task, different repo) but should be *posted* after
  P001 is citable on arXiv, so its "confirms locality" claim is not resting on an
  unpublished result.
- R001 (arXiv v2 of the core implicit-EM paper) is **already updated to v2**, so the
  framing/notation dependency for P001 is satisfied. P001 is the natural next stage
  in the implicit-EM line; it is mostly theoretical.

### Stochastic-map arm — decided IN

Include the row-stochastic-`W2` arm (§7.9). Rationale: it is the one preservation
class with no experiment AND the mechanistic bridge to the attention project —
attention's value path is row-stochastic, so a positive result (partial
conditioning survives) predicts that at-site volume control transports to attention
while conditioning does not. Report either way; a null still populates the taxonomy
cell. Risk: row-normalization may hurt head accuracy enough to confound the probe.

### Formal-structure decisions (for §3)

- Write §3 as **numbered environments**: Lemma 1 (simplex invariant, ½ bound),
  Proposition 1 (private-parameter data-constant curvature), Proposition 1'
  (smooth monotone kernels / Softplus; ReLU a.e. + absorbing-state caveat),
  Proposition 2 (composition → `sigma_max(W2)^2` scaling), Proposition 3
  (informal: shared lower map makes Q non-separable = classical shared-parameter
  EM failure). Full proofs to an appendix.
- Add a **boxed "Scope of the claim"** note after Prop 2: sufficient conditions +
  demonstrated failures, explicitly NOT iff, no necessity, no convergence proof.
  This is where the top-ranked "iff temptation" risk is guarded.
- **Derive** the "parameter sharing in disguise" punchline (Prop 3) before the
  classical-EM table; the table is a map, not the argument.
- LayerNorm: prefer the clean theorem WITHOUT it; note the empirical curvature
  measurements are on the actual model and absorb its (bounded) effect.

### New scaffolding added to the outline (track these as writing tasks)

- **§0 Notation** block — fixes distance/energy sign convention once
  (d = energy, lower = better; r = softmax(-d); dL_LSE/dd_j = +r_j).
- **§2 Related Work** as its own section (was scattered): Dempster-Laird-Rubin,
  Neal & Hinton, Xu & Jordan, Böhning/Popoviciu, Ng & Jordan, Frankle et al. /
  Entezari (LMC + permutation alignment), Hinton 2006 (DBN lineage).
- **§7.0 Predictions ledger** (prediction → measured → verdict) — pre-empts
  post-hoc storytelling; includes the visible λ≠ratio correction (predicted ~1000×,
  measured 30-70×).
- **§9 split**: Threats to Internal Validity (single probe → add kNN/MLP probe;
  curvature batch/iteration sensitivity; ½-diagnostic only shown at one site;
  stop-gradient as clean causal knob) vs Scope Limitations (MNIST, small nets).
- **Promoted to MAIN results** (were "optional"): λ-resolution curve (central
  figure, x = measured CE/EM ratio), basin/LMC analysis (Hungarian-aligned barrier
  plot). Matches the proposal's own ranking.

### Feasibility

User has a local RTX 3080 Ti (12 GB + up to 76 GB shared, shared is slow but
usable). All remaining runs — Adam arm, λ-curve, basin, depth sweep, stochastic-map,
Fashion-MNIST — are feasible locally. No compute blocker on scope.

### Depth sweep — where it lives

The depth sweep (2-6 stacked EM sites, per-layer ½-saturation) earns the
diagnostic-generality claim (contribution 4). Candidate to **defer to P004
(Layer-Wise Implicit EM)** as the empirical extension rather than bloat P001.
Decide during drafting; parked in P001's Required-Work as deferrable.

