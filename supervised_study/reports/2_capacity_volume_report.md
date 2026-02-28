# Experiment 2: Capacity and Volume Control

## Context

Experiment 1 established that EM at an intermediate layer requires volume control: both variance (anti-collapse) and decorrelation (anti-redundancy) together. Those results used a single hidden dimension (25 units). This experiment asks how the interaction between volume control and model capacity behaves as hidden dimension varies.

## Design

Two configurations compared across five hidden dimensions:

| Config | NLS | Var | Decorr |
|---|:---:|:---:|:---:|
| Baseline | — | — | — |
| NLS + Var + Decorr | ✓ | ✓ | ✓ |

Hidden dimensions: 16, 25, 36, 49, 64. Training: MNIST, 50 epochs, Adam lr=0.001, λ_reg=0.001, 5 seeds (42–46). All summary metrics averaged over the last 10 epochs (41–50).

## Results

| Hidden | Config | Dead Units | Dead Frac | Min Var | Redundancy | Accuracy |
|---|---|---|---|---|---|---|
| 16 | baseline | 1.0 ± 0.9 | 6.3% | 0.097 ± 0.119 | 11.0 ± 2.2 | 95.19% ± 0.19% |
| 16 | nls_var_tc | 0.0 ± 0.0 | 0% | 5.22 ± 1.09 | 9.1 ± 0.9 | 95.32% ± 0.21% |
| 25 | baseline | 1.4 ± 0.5 | 5.6% | 0.000 ± 0.000 | 19.0 ± 1.3 | 96.07% ± 0.07% |
| 25 | nls_var_tc | 0.0 ± 0.0 | 0% | 7.76 ± 1.21 | 13.8 ± 0.5 | 96.35% ± 0.12% |
| 36 | baseline | 2.4 ± 1.9 | 6.7% | 0.030 ± 0.059 | 31.7 ± 5.1 | 96.63% ± 0.14% |
| 36 | nls_var_tc | 0.0 ± 0.0 | 0% | 9.51 ± 0.10 | 21.1 ± 1.5 | 96.60% ± 0.10% |
| 49 | baseline | 2.5 ± 0.9 | 5.0% | 0.000 ± 0.000 | 48.1 ± 3.6 | 97.04% ± 0.09% |
| 49 | nls_var_tc | 0.0 ± 0.0 | 0% | 12.65 ± 1.55 | 29.7 ± 0.5 | 96.87% ± 0.16% |
| 64 | baseline | 4.1 ± 1.3 | 6.3% | 0.000 ± 0.000 | 69.0 ± 7.2 | 97.29% ± 0.06% |
| 64 | nls_var_tc | 0.0 ± 0.0 | 0% | 15.33 ± 0.92 | 41.2 ± 0.8 | 97.00% ± 0.09% |

![Redundancy vs Capacity](capacity_redundancy.png)

![Volume Control Benefit vs Capacity](capacity_accuracy.png)

## Findings

### 1. Dead units appear at every capacity

Every baseline model has dead units, regardless of hidden dimension. The dead unit fraction is remarkably stable: 5.0%–6.7% across a 4× range of hidden dimensions. At 16 units, ~1 unit dies. At 64, ~4 die. This is a property of the optimization dynamics (ReLU + Adam + CE), not the architecture width.

Volume control eliminates dead units completely at every hidden dimension. Zero dead units, zero variance, across all seeds and sizes.

### 2. Redundancy scales linearly with capacity

Baseline redundancy grows roughly linearly: 11 → 19 → 32 → 48 → 69 as hidden dim increases from 16 to 64. Wider models waste more capacity on correlated features. Volume control reduces redundancy at every size, but both curves grow — VC doesn't eliminate the scaling, it reduces the slope.

The redundancy gap widens with capacity. At hidden dim 16, the difference is small (11.0 vs 9.1). At 64, it's substantial (69.0 vs 41.2). The decorrelation penalty has more work to do as the number of off-diagonal pairs grows quadratically.

### 3. Volume control tightens variance across seeds

At every hidden dimension, the nls_var_tc config has tighter standard deviations on redundancy than the baseline (e.g., 0.8 vs 7.2 at hidden dim 64). Volume control produces more reproducible representations. The baseline's structural metrics vary substantially across seeds; the VC-regularized models converge to similar geometry every time.

### 4. Accuracy benefit depends on capacity and λ

Volume control improves accuracy at hidden dim 16 and 25 (+0.13%, +0.28%), is neutral at 36, and slightly hurts at 49 and 64 (−0.17%, −0.29%). The accuracy gain peaks at 25 and crosses zero around 36.

However, this comparison uses a fixed λ_reg=0.001 at all sizes. The calibration sweep (run separately) showed that optimal λ shifts with capacity: λ=0.001 is appropriate for hidden dim 16–25, but hidden dim 64 prefers λ=0.0001. At that lighter regularization, the sweep showed a small accuracy gain (+0.11%) even at 64 units. The accuracy crossover in this experiment is at least partly a λ calibration artifact, not a fundamental limitation.

The structural benefits — zero dead units, lower redundancy, tighter variance — hold regardless of λ calibration. The accuracy benefit requires matching λ to capacity.

### 5. ~25 units may be a natural capacity for MNIST

The accuracy benefit peaks at hidden dim 25, which is roughly the number of distinct digit variants (a few per class). At this capacity, each unit functions as a prototype for a digit variant, and volume control ensures every prototype is alive and distinct. This is exactly what a mixture model with the right number of components should do.

Beyond 25, the baseline uses extra units differently — not as additional prototypes but as cooperative boundary refiners. Multiple units with correlated activations trace finer decision boundaries. The decorrelation penalty treats this correlation as redundancy and penalizes it, even though it serves a different function (cooperative approximation rather than wasteful duplication).

## Interpretation

The capacity experiment reveals two regimes:

**Capacity-constrained (≤25 units):** Every unit must be a distinct, useful feature. Dead units are costly — a dead unit at hidden dim 16 is 6.25% of capacity lost. Redundant units are wasteful — two units doing the same thing means one fewer distinct feature. Volume control helps both accuracy and structure by ensuring the limited capacity is fully utilized.

**Capacity-abundant (≥36 units):** The network has more units than it needs for distinct prototypes. Extra units are used for fine-grained boundary refinement via correlated activation patterns. Dead units are cheap — the network routes around them. The decorrelation penalty at λ=0.001 is too aggressive, preventing the cooperative correlation that boundary refinement requires.

Overparameterization and volume control are two strategies for the same problem. Overparameterization tolerates dead and redundant units by providing enough spare capacity. Volume control keeps every unit alive and distinct. The first wastes capacity but is unconstrained. The second is efficient but imposes geometric constraints that become restrictive at high capacity.

## Limitations

**Fixed λ across sizes.** The sweep data shows that optimal λ depends on capacity. This experiment confounds "does VC help" with "is λ calibrated." Per-size λ would give a fairer accuracy comparison.

**Five seeds.** Sufficient for the structural claims (dead units, redundancy) which have small variance. The accuracy differences at large hidden dims are within overlapping error bars and would benefit from more seeds.

**Single dataset.** MNIST has ~25 natural components. The capacity threshold where VC stops helping accuracy likely depends on the intrinsic dimensionality of the data.

## Future Work

**Independent λ_var and λ_tc.** The variance and decorrelation penalties likely have different sensitivity profiles. The variance penalty has a sharp threshold — just enough to prevent dead units. The decorrelation penalty constrains representation geometry and may need to be relaxed at high capacity. Sweeping them independently would clarify which term causes the accuracy cost at large hidden dims and whether a light variance penalty alone can eliminate dead units without restricting boundary refinement.

**Per-capacity λ calibration.** The sweep data suggests optimal λ scales inversely with capacity. A principled scaling rule (e.g., λ_tc ∝ 1/hidden_dim) might maintain the structural benefits across sizes without the accuracy cost.

**Hierarchical residual architecture.** If ~25 components is the natural prototype count for MNIST, a two-layer design could separate the mixture structure from boundary refinement: learn prototypes with VC in the first layer (capturing "which component am I near"), then learn fine-grained corrections in a second layer without VC (capturing "where exactly within that component's region"). This would be a hierarchical mixture — coarse EM structure with residual supervised refinement — and a natural extension of the implicit EM framework to deeper networks.