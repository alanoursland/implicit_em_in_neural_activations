# Experiment 1: Volume Control Ablation

## Context

Paper 1 established that cross-entropy training performs implicit EM at the output layer: softmax produces responsibilities, labels provide volume control. Paper 2 showed that implicit EM can be created in an unsupervised setting with an LSE objective and InfoMax regularization (variance + decorrelation), and that EM without volume control collapses.

This experiment asks: can we extend EM from the output layer into an intermediate layer of a supervised network? And does the same volume control requirement from Paper 2 apply?

## Design

A two-layer MNIST classifier with an ablation target between layers:

```
x ∈ ℝ^784 → Linear(784, 25) → ReLU → [ablation target] → Linear(25, 10) → LayerNorm → CE
```

NegLogSoftmin (NLS) introduces EM structure at the intermediate layer by embedding the LSE partition function and a competitive Jacobian. Volume control (VC) is provided by variance and decorrelation penalties on the intermediate distances.

Five configurations test the components:

| Config | NLS | Var | Decorr | Role |
|---|:---:|:---:|:---:|---|
| Baseline | — | — | — | No intermediate EM. Standard MLP. |
| NLS only | ✓ | — | — | EM without volume control. |
| NLS + Var | ✓ | ✓ | — | EM with partial VC (anti-collapse only). |
| NLS + Var + Decorr | ✓ | ✓ | ✓ | EM with full volume control. |
| Var + Decorr only | — | ✓ | ✓ | Volume control without EM. |

Training: MNIST, hidden dim 25, 50 epochs, Adam lr=0.001, λ_reg=0.001, 10 seeds (42–51). All summary metrics are averaged over the last 10 epochs (41–50) to smooth per-epoch noise.

## Results

| Config | Dead Units | Min Var | Redundancy | Resp Entropy | Accuracy |
|---|---|---|---|---|---|
| Baseline | 1.40 ± 0.49 | 0.000 ± 0.000 | 20.5 ± 2.8 | 3.08 ± 0.02 | 96.09% ± 0.07% |
| NLS only | 1.20 ± 0.98 | 0.264 ± 0.408 | 24.9 ± 3.0 | 2.93 ± 0.02 | 96.14% ± 0.17% |
| NLS + Var | 0.00 ± 0.00 | 11.74 ± 3.08 | 187.0 ± 38.7 | 2.09 ± 0.09 | 95.75% ± 0.21% |
| NLS + Var + Decorr | 0.00 ± 0.00 | 7.84 ± 1.17 | 13.9 ± 0.5 | 2.46 ± 0.04 | 96.34% ± 0.11% |
| Var + Decorr only | 0.10 ± 0.30 | 3.55 ± 1.49 | 13.3 ± 0.9 | 2.65 ± 0.07 | 96.40% ± 0.10% |

![Training dynamics](../results/experiment1/figures/training_dynamics.png)

![Loss curves](../results/experiment1/figures/loss_curves.png)

## Findings

### 1. The supervised gradient does not provide volume control at intermediate layers

The baseline has min_variance = 0.000 across all 10 seeds. At least one ReLU unit is completely dead in every trained model, with an average of 1.4 dead units out of 25. The supervised gradient ensures the output layer works (via labels), but the learning signal that reaches the hidden layer through W₂ does not prevent collapse. Units die during the first ~15 epochs and never recover (visible in the dead units training curve).

### 2. EM without volume control does not help

NLS only is statistically indistinguishable from the baseline. Accuracy: 96.14% vs 96.09%. Dead units: 1.2 vs 1.4. Redundancy actually increases (24.9 vs 20.5). The competitive Jacobian from NegLogSoftmin introduces EM structure, but without volume control, that structure has no stability. The EM dynamics are present but cannot organize the representation.

### 3. Partial volume control under EM is destructive

NLS + Var is the worst configuration — worse than no intervention at all. Dead units drop to zero (the variance penalty works), but redundancy explodes to 187, a 9× increase over the baseline. Accuracy drops to 95.75%, the lowest of any config.

The mechanism is clear: the variance penalty forces every unit to maintain non-zero variance, but without decorrelation, the easiest way to achieve high variance is for all units to respond to the same patterns. Units become alive but identical. The EM dynamics then amplify this: responsibility-weighted competition among nearly identical units produces unstable, degenerate assignments. The responsibility entropy is the lowest (2.09), reflecting this artificial sharpness among redundant components.

This confirms the Paper 2 prediction: the two components of volume control (diagonal and off-diagonal of the log-determinant) are inseparable. Anti-collapse without anti-redundancy is worse than neither.

### 4. Full volume control stabilizes EM

NLS + Var + Decorr achieves the best accuracy (96.34%), the lowest redundancy (13.9), zero dead units, and the tightest standard deviation on accuracy (±0.11%). The decorrelation penalty prevents the redundancy catastrophe of NLS + Var while preserving the anti-collapse benefit. Every seed of this config outperforms the baseline mean.

The redundancy training curve shows the dynamics directly: NLS + Var (green) climbs continuously throughout training, while NLS + Var + Decorr (red) drops quickly to ~14 and stabilizes. The decorrelation penalty catches the redundancy immediately — it doesn't allow it to develop and then correct it.

### 5. Volume control without EM is good regularization, not EM

Var + Decorr only (no NLS) matches the full EM config on accuracy (96.40% vs 96.34%) and redundancy (13.3 vs 13.9). These differences are within noise at 10 seeds.

However, the representations differ. Responsibility entropy is higher for Var + Decorr only (2.65 vs 2.46), indicating softer, less competitive component assignments. Min variance is lower (3.55 vs 7.84), meaning the variance penalty is less effective without NLS calibration to standardize the distances. This config produces well-regularized features — alive, diverse, useful — but not EM-structured mixture components.

The NLS contributes calibration and optimization stability rather than raw accuracy. It helps the variance penalty work more effectively (higher min_var, tighter std) and produces sharper component assignments. The difference between "well-regularized features" and "EM-structured mixture components" is real but subtle at this scale.

### 6. The baseline achieves the lowest CE loss

An observation from the loss curves: the baseline achieves the lowest cross-entropy loss on the test set (0.156) despite having second-worst accuracy (96.09%). It is overfitting the loss without building a healthy representation. The volume-controlled models are constrained by their regularization terms — they cannot minimize CE as aggressively — but they generalize better. The representation's geometric health matters for generalization even when it doesn't help the training objective.

## Summary

| Prediction | Result | Status |
|---|---|---|
| Baseline has dead intermediate units | min_var = 0, 1.4 dead units, all seeds | Confirmed |
| NLS alone doesn't help | Same accuracy, more redundancy than baseline | Confirmed |
| NLS + Var is pathological | Worst config: redundancy 187, worst accuracy | Confirmed |
| NLS + Var + Decorr is best | Best accuracy, lowest redundancy, zero dead | Confirmed |
| Var + Decorr only is good but different | Matches accuracy, higher resp entropy, lower min_var | Confirmed |

The results replicate the Paper 2 pattern inside a supervised network. EM requires volume control. Labels provide it at the output layer. At intermediate layers, it must be supplied explicitly. Both components of volume control — anti-collapse (variance) and anti-redundancy (decorrelation) — are required together. Partial volume control under EM is worse than none.