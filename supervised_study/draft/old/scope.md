# Scope

## Pattern

Paper 1 derived a theorem. Paper 2 tested it in the unsupervised regime. This paper tests it in the supervised regime.

Paper 1 scope: one identity, one interpretation, one unification. No experiments.

Paper 2 scope: one question (is the theory prescriptive?), one model, five predictions. All confirmed.

This paper scope: one question, one model, one ablation.

## The Question

Does implicit EM theory correctly predict the behavior of intermediate layers in supervised networks?

The theory says:
- EM arises only where exponentiation + normalization occurs
- Standard intermediate layers (ReLU, Softplus) lack EM structure
- Volume control is needed wherever EM operates
- Labels provide volume control at the output but not at intermediate layers

These are predictions. We test them.

## The Model

A two-layer supervised network with an ImplicitEM layer, derived from theory. Minimal. MNIST. Same methodology as Paper 2: build what the theory prescribes, nothing more, nothing less.

## The Ablation

Remove components of the ImplicitEM layer. Observe whether the predicted failure modes appear. Same structure as Paper 2's ablation.

Predictions:
- LSE alone collapses the intermediate representation
- Variance prevents dead components
- Decorrelation prevents redundancy
- Full ImplicitEM produces structured mixture components

If confirmed: the theory extends to supervised networks. Intermediate layers need volume control even with supervised gradients flowing through.

If not confirmed: the theory's scope is narrower than claimed. Supervised gradients provide something the theory doesn't account for. Also a result.

## What Is In Scope

- The ImplicitEM layer (derivation from theory, implementation)
- The ablation (six configs, three primary metrics, same format as Paper 2)
- Intermediate layer health (dead units, redundancy, responsibility entropy)
- Weight visualization (do intermediate features show mixture structure?)
- Classification accuracy (does volume control help, hurt, or not affect output?)

## What Is Out of Scope

- Formal analysis of NegLogSoftmin Jacobian (supporting material, not the contribution)
- Kernel comparison (Softplus vs ReLU vs Gaussian — future work)
- Scaling beyond MNIST (acknowledged limitation, same as Paper 2)
- LLM activations (acknowledged limitation, same as Paper 2)
- Multiple ImplicitEM layers / deep stacking (future work)
- Training dynamics (report if interesting, but not a primary result)
- The claim that heuristics are implicit volume control (discussion paragraph, not tested)
- Single-layer CE analysis (setup context, not an experiment)
- Comparison with Paper 2's unsupervised model (brief if illuminating, not a primary result)

## The Contribution

If predictions hold: implicit EM theory correctly predicts intermediate layer behavior in supervised networks. Volume control, derived from mixture model theory, is needed at intermediate layers even when supervision is present. The ImplicitEM layer is a principled solution.

If predictions fail: the boundary of implicit EM theory is more constrained than Paper 1 claimed. The supervised regime behaves differently from the unsupervised regime in ways the theory does not capture. This constrains future work.

Either way: the experiment tests the theory. That is the contribution.