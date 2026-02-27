# Ablation Report: Volume Control in Supervised Networks

## Purpose

Test whether implicit EM theory correctly predicts the behavior of intermediate layers in supervised networks. Specifically: does the supervised gradient prevent collapse? Does volume control fix it? Do both components (variance + decorrelation) need to work together?

## Setup

- **Architecture:** Linear(784, 25) → ReLU → [ablation target] → Linear(25, 10) → LayerNorm → CE
- **Dataset:** MNIST, 60K train / 10K test
- **Hidden dim:** 25
- **Batch size:** 128
- **Epochs:** 40
- **Optimizer:** Adam, lr=0.001
- **λ_reg:** 0.001 (calibrated via sweep, see sweep_report.md)
- **Seeds:** 42, 43, 44

## Configurations

| # | Config | NegLogSoftmin | Variance | Decorrelation | What It Tests |
|---|--------|:---:|:---:|:---:|---|
| 1 | baseline | — | — | — | Standard MLP. Does supervision prevent collapse? |
| 2 | nls_only | ✓ | — | — | Does competitive Jacobian alone help? |
| 3 | nls_var | ✓ | ✓ | — | Does anti-collapse without anti-redundancy help? |
| 4 | nls_var_tc | ✓ | ✓ | ✓ | Full ImplicitEM layer. |
| 5 | var_tc_only | — | ✓ | ✓ | Volume control without competitive Jacobian. |

## Results

| Config | Min Variance | Redundancy | Resp Entropy | Accuracy |
|--------|-------------|-----------|--------------|----------|
| baseline | 0.000 ± 0.000 | 18.9 ± 0.8 | 3.091 ± 0.006 | 96.17% ± 0.07% |
| nls_only | 0.293 ± 0.415 | 25.2 ± 2.7 | 2.931 ± 0.010 | 96.11% ± 0.18% |
| nls_var | 12.168 ± 1.193 | 178.4 ± 55.7 | 2.104 ± 0.135 | 95.80% ± 0.25% |
| nls_var_tc | 6.747 ± 1.376 | 13.9 ± 0.6 | 2.459 ± 0.055 | 96.51% ± 0.06% |
| var_tc_only | 3.138 ± 0.770 | 13.6 ± 1.2 | 2.663 ± 0.096 | 96.37% ± 0.10% |

## Findings

### 1. Supervised gradient does not prevent collapse

The baseline has min_variance = 0.0 across all three seeds. Dead ReLU units exist at the intermediate layer of a trained, converged supervised network. The supervised gradient does not provide anti-collapse at intermediate layers.

This confirms the theory's central prediction.

### 2. NegLogSoftmin alone has minimal effect

Config 2 (nls_only) is statistically indistinguishable from the baseline in accuracy (96.11% vs 96.17%). Min variance is 0.29 with high variance across seeds (±0.41), meaning some seeds still have dead units. Redundancy actually increases from 18.9 to 25.2.

The competitive Jacobian from NegLogSoftmin does not prevent collapse or reduce redundancy on its own. Without volume control, the EM dynamics have no structural benefit.

### 3. Variance without decorrelation is harmful

Config 3 (nls_var) is the most informative result. Dead units are fixed (min_var = 12.2). But redundancy explodes to 178.4 — a 9.4× increase over baseline. Accuracy drops to 95.80%, the worst of all configs.

The variance penalty forces every unit to have high activation variance. Without decorrelation, units achieve this by becoming copies of each other — all responding strongly to the same patterns. The representation is alive but degenerate. Anti-collapse without anti-redundancy is worse than no regularization at all.

This confirms the theory's prediction that both components of volume control (diagonal and off-diagonal of the log-determinant) are required. Neither is sufficient alone.

### 4. Full volume control (var + tc) is needed

Config 4 (nls_var_tc) achieves the best accuracy (96.51%), lowest redundancy (13.9), and healthy min variance (6.7). Every seed of config 4 beats every seed of the baseline. The improvement is small (+0.34%) but consistent and non-overlapping in range.

Both components contribute: variance prevents dead units, decorrelation prevents the redundancy that variance alone creates. Together they produce a structured intermediate representation that improves classification.

### 5. Volume control does most of the work

Config 5 (var_tc_only) — volume control without NegLogSoftmin — achieves 96.37% accuracy, redundancy 13.6, min variance 3.1. This is close to the full ImplicitEM result (config 4: 96.51%, 13.9, 6.7).

NegLogSoftmin adds a small increment: +0.14% accuracy and higher min variance (6.7 vs 3.1). The competitive Jacobian helps but is not the primary driver. Volume control is doing the heavy lifting.

### 6. Responsibility entropy reflects competition

Baseline resp_entropy is 3.09 — broadly distributed responsibilities. All configs with NegLogSoftmin or volume control have lower entropy (2.1-2.9), indicating sharper competition and component specialization. Config 3 (nls_var) has the lowest entropy (2.10) but this reflects redundancy, not healthy competition — many identical units create artificially sharp assignments.

## Summary of Predictions vs Results

| Prediction | Result | Status |
|-----------|--------|--------|
| Supervised gradient doesn't prevent collapse | min_var = 0 in baseline | Confirmed |
| Volume control needs both variance and decorrelation | Variance alone explodes redundancy to 178.4 | Confirmed |
| Full volume control produces structured representations | Best accuracy, lowest redundancy | Confirmed |
| NegLogSoftmin provides competitive gradient structure | Minimal effect alone, small increment with VC | Partially confirmed |
| Volume control without NLS should whiten, not cluster | var_tc_only performs nearly as well as full ImplicitEM | Partially disconfirmed |

## Interpretation

The theory is substantially confirmed. The central predictions hold:

1. Intermediate layers lack volume control from supervision.
2. Both components of volume control (anti-collapse + anti-redundancy) are needed.
3. Full volume control improves both representation health and classification accuracy.

The nuance: NegLogSoftmin's contribution is smaller than expected. Volume control alone (config 5) provides most of the benefit. The competitive Jacobian adds a modest increment. The theory predicted that EM dynamics and volume control are both necessary. The data suggests volume control is necessary and EM dynamics are helpful but secondary.

The nls_var result (config 3) is the strongest piece of evidence. It shows that the two components of volume control are not independently useful — they must work together. This is exactly what the log-determinant formulation predicts: the determinant requires both diagonal (variance) and off-diagonal (correlation) terms to maintain volume.

## Limitations

- MNIST only. Single dataset, single task.
- 40 epochs. Sweep showed regularization loss still falling. 100 epochs may change relative rankings.
- Single hidden dim (25). Sweep showed capacity dependence — results differ at 16 vs 64.
- ReLU activation. Softplus would eliminate the collapse failure mode entirely.
- Three seeds. Small sample for statistical claims.

## Next Steps

1. Run at 100 epochs to check convergence.
2. Generate weight visualizations for all five configs.
3. Consider whether the capacity dependence (sweep data across hidden dims) belongs in the paper.
4. Decide paper framing given the partial disconfirmation of NLS importance.