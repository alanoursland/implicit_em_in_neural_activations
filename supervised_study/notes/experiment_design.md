# Experiment Design: Volume Control in Supervised Networks

## Research Questions

1. Does the supervised gradient provide volume control at intermediate layers?
2. Are both components of volume control (anti-collapse and anti-redundancy) needed together?
3. How does model capacity interact with the need for volume control?

## Stories This Paper Tells

**Story 1: The volume control ablation.** Intermediate layers in supervised networks lack volume control. The log-determinant decomposes into two inseparable components — variance (anti-collapse) and decorrelation (anti-redundancy). Variance alone is worse than nothing. Both together produce the best representation.

**Story 2: Capacity vs. structure.** Overparameterization and volume control are two solutions to the same problem. Wide networks tolerate dead and redundant units by routing around them. Volume control keeps every unit alive and distinct. The first wastes capacity; the second uses it efficiently.

---

## Base Architecture

All experiments share:

```
x ∈ ℝ^784 (flattened MNIST)
    → Linear(784, K) → ReLU           [distances]
    → [ablation target]                [calibration + volume control]
    → Linear(K, 10) → LayerNorm(10)   [classification head]
    → CrossEntropy(·, labels)          [supervised loss]
```

ReLU activation. This preserves the collapse failure mode that volume control addresses. Softplus would eliminate dead units by construction, removing the phenomenon under study. Consistent with Paper 2.

## Shared Training Protocol

- **Dataset:** MNIST, 60K train / 10K test
- **Batch size:** 128
- **Epochs:** 50
- **Optimizer:** Adam, lr = 0.001
- **λ_reg:** 0.001 (overall weight for auxiliary loss, calibrated via sweep)

---

## Experiment 1: Volume Control Ablation

### Purpose

Test whether both components of volume control are needed at intermediate layers, even with supervised gradients flowing through. The centerpiece result is that variance without decorrelation is actively harmful.

### Configurations

| # | Config | NLS | Var | TC | What It Tests |
|---|--------|:---:|:---:|:---:|---|
| 1 | baseline | — | — | — | Does supervision prevent collapse? |
| 2 | nls_only | ✓ | — | — | Does the competitive Jacobian alone help? |
| 3 | nls_var | ✓ | ✓ | — | Is anti-collapse sufficient without anti-redundancy? |
| 4 | nls_var_tc | ✓ | ✓ | ✓ | Full ImplicitEM layer |
| 5 | var_tc_only | — | ✓ | ✓ | Volume control without calibration |

These map directly to the existing `configs.py`. No code changes needed for the configs.

### Parameters

- **Hidden dim:** 25
- **Seeds:** 10 (seeds 42–51)
- **λ_reg:** 0.001

Hidden dim 25 chosen because the capacity is tight enough that dead units cost measurable accuracy, giving the clearest signal. Confirmed by sweep.

### Predictions

| Prediction | Source |
|---|---|
| Baseline has dead units (min_var = 0) | Supervision doesn't control intermediates |
| nls_only ≈ baseline | Jacobian alone insufficient without VC |
| nls_var is worst config | Anti-collapse without anti-redundancy is destructive |
| nls_var has highest redundancy | Variance forces activity; without TC, units copy each other |
| nls_var_tc is best on accuracy + structure | Both VC components needed together |
| var_tc_only close to nls_var_tc | VC does the heavy lifting; NLS adds modest increment |

### Metrics

Evaluated on full test set at every epoch:

- **Dead units:** Count of components with Var(d_j) < 0.01
- **Min variance:** Minimum per-component variance (continuous measure of collapse)
- **Redundancy:** ‖Corr(d) − I‖²_F off-diagonal
- **Responsibility entropy:** E_x[H(softmin(d(x)))]
- **Test accuracy:** Classification accuracy on MNIST test set
- **CE loss:** Cross-entropy on test set
- **Reg loss:** Volume control regularization loss on test set

### Evaluation Protocol

Summary tables report the **mean over the last 10 epochs** (epochs 41–50) for every metric — accuracy, dead units, min variance, redundancy, responsibility entropy, CE loss, reg loss. This smooths out per-epoch noise from batch ordering without selecting on the evaluation set (no "best epoch" or "best accuracy" cherry-picking).

Per-epoch training curves show raw values. The noise is honest and shows convergence behavior. Summary tables use the smoothed values.

If a config hasn't converged by epoch 40, averaging epochs 41–50 captures a still-moving model. This is a fair comparison across configs. Instability at convergence is itself informative — it's a property of the configuration, not a measurement artifact.

### Per-Epoch Tracking

Log all metrics above at every epoch. This produces the training dynamics curves. The dead-unit-over-training plot is particularly important: it should show baseline units dying and staying dead, while nls_var_tc units remain alive from the start.

### Deliverables

1. **Ablation table:** Mean ± std across 10 seeds for all metrics, all 5 configs.
2. **Training dynamics figure:** Per-epoch dead units, redundancy, and test accuracy for all 5 configs.
3. **Loss curves figure:** CE loss and reg loss over training for all 5 configs.

### Total runs: 50

---

## Experiment 2: Capacity and Volume Control

### Purpose

Show that volume control's benefit depends on capacity. When capacity is tight, every unit matters and volume control helps substantially. When capacity is abundant, the network tolerates dead and redundant units. Overparameterization is a brute-force substitute for principled volume control.

### Design

Two configs compared across five hidden dimensions:

| Config | NLS | Var | TC |
|---|:---:|:---:|:---:|
| baseline | — | — | — |
| nls_var_tc | ✓ | ✓ | ✓ |

| Hidden dim | 16 | 25 | 36 | 49 | 64 |
|---|---|---|---|---|---|

### Parameters

- **Seeds:** 5 (seeds 42–46)
- **λ_reg:** 0.001 for nls_var_tc
- **Epochs:** 50

### Metrics

Same as Experiment 1 (same per-epoch tracking, same last-10-epoch averaging for summary tables), plus:

- **Dead unit count** (not just min variance — we want to say "8 of 64 units died")
- **Dead unit fraction** (dead units / hidden dim — normalizes across sizes)

### Predictions

| Prediction | Source |
|---|---|
| Dead units at all hidden dims (baseline) | Supervision doesn't scale with capacity |
| Redundancy scales roughly linearly with hidden dim (baseline) | More units, more redundancy |
| Accuracy benefit of VC largest at small hidden dim | Every unit matters when capacity is tight |
| Accuracy benefit of VC shrinks toward noise at large hidden dim | Network routes around dead units |
| VC halves redundancy at every hidden dim | Anti-redundancy is capacity-independent |
| Dead unit fraction roughly constant (baseline) | Collapse rate is a property of the optimization, not the width |

### Deliverables

1. **Capacity table:** For each hidden dim, baseline vs nls_var_tc on accuracy, dead units, dead unit fraction, redundancy. Mean ± std across 5 seeds.
2. **Capacity figure:** Accuracy gain (nls_var_tc − baseline) vs hidden dim. Should show decreasing benefit with increasing capacity.
3. **Redundancy figure:** Redundancy vs hidden dim for both configs. Should show linear scaling for baseline, flatter for nls_var_tc.

### Total runs: 50

---

## Experiment 3: Optimization Dynamics

### Purpose

Paper 2 found that the implicit EM objective produces unusual optimization behavior: SGD is learning-rate insensitive across three orders of magnitude, Adam offers no advantage, and lower loss does not correspond to better features. These were interpreted as evidence that responsibility-weighted gradients naturally condition the landscape.

This experiment tests whether those properties survive when a supervised CE loss is added. The total loss is now CE + λ·(var + tc). The CE gradient is a standard supervised signal with no EM structure. If the optimization anomalies persist, the volume control terms dominate the landscape. If they disappear, the supervised component introduces the ill-conditioning that Adam is designed to handle.

### Design

One config (nls_var_tc), two optimizers, four learning rates each:

| Optimizer | Learning rates |
|---|---|
| SGD | 0.0001, 0.001, 0.01, 0.1 |
| Adam | 0.0001, 0.001, 0.01, 0.1 |

### Parameters

- **Hidden dim:** 25
- **Seeds:** 3 (seeds 42–44)
- **λ_reg:** 0.001
- **Epochs:** 50

### Metrics

All summary values use last-10-epoch averaging (epochs 41–50), same as Experiments 1 and 2:

- Test accuracy
- CE loss
- Reg loss
- Per-epoch loss curves (raw values, for the loss trajectory figure)

### Predictions

| Prediction | If Paper 2 pattern holds | If supervision disrupts it |
|---|---|---|
| SGD lr sensitivity | Insensitive (92–94% across 1000×) | Sensitive (low lr fails) |
| Adam vs SGD | Comparable accuracy | Adam clearly better |
| Loss vs features | Adam lower loss, same accuracy | Lower loss = better accuracy |

Either outcome is informative. Replication extends the Paper 2 finding to supervised settings. Failure to replicate constrains it — the well-conditioned landscape may be specific to purely unsupervised EM objectives.

### Deliverables

1. **Optimizer table:** Final accuracy and loss for each optimizer × lr combination. Mean ± std across 3 seeds.
2. **Loss curves figure:** Loss over training for SGD (4 lr) and Adam (4 lr), same format as Paper 2 Figure 3.

### Total runs: 24

---

## What Is Not In This Experiment

**No LSE auxiliary loss.** The original design docs included configs with an explicit LSE auxiliary loss term. The implementation dropped it. NegLogSoftmin provides the LSE structure in the forward pass (the partition function and competitive Jacobian). Volume control regularizes the distances directly. This is a simpler and cleaner design. The EM dynamics come from NegLogSoftmin shaping the supervised gradient, not from a separate unsupervised loss.

**No Softplus activation.** The design docs argued for Softplus over ReLU. ReLU is used because it preserves the collapse failure mode. Softplus would make min_variance always positive, eliminating the dead unit phenomenon. A Softplus comparison is natural follow-up work but not needed for the current claims.

**No weight visualization as primary evidence.** Weight visualizations are generated but the quantitative metrics (dead units, redundancy, accuracy) carry the argument. The visual differences between configs are real but subtle.

**No λ sweep in the paper.** The sweep was done to calibrate λ_reg = 0.001. The sweep data informs the capacity story but the paper reports the two-config comparison, not the full sweep grid.

---

## Implementation Notes

### Code changes needed

1. **Training loop:** Add per-epoch metric logging (dead units, redundancy, min variance on test set). Currently only logs loss and accuracy per epoch.

2. **Metrics:** Add `dead_unit_count(a, threshold=0.01)` returning an integer count. The existing `min_variance` returns a float; we want both.

3. **Run scripts:** Two scripts or one script with experiment flag:
   - Experiment 1: 5 configs × 10 seeds × 50 epochs at hidden_dim=25
   - Experiment 2: 2 configs × 5 hidden_dims × 5 seeds × 50 epochs

4. **Data dir:** Set `data_dir="./data"` and `use_gpu_cache=False` for non-Windows environments.

### Runtime estimate

- Per run at hidden_dim=25: ~2–3 min on CPU, ~30s on GPU
- Experiment 1 (50 runs): ~2.5 hrs CPU / ~25 min GPU
- Experiment 2 (50 runs): ~3 hrs CPU / ~30 min GPU (larger hidden dims cost more)
- Experiment 3 (24 runs): ~1 hr CPU / ~12 min GPU
- Total: ~6.5 hrs CPU / ~1.1 hr GPU

### Output structure

```
results/
  experiment1/
    ablation_table.csv
    raw_results.json
    training_curves.json        # per-epoch metrics for dynamics plots
    figures/
      ablation_table.png
      training_dynamics.png     # dead units, redundancy, accuracy over epochs
      loss_curves.png
  experiment2/
    capacity_table.csv
    raw_results.json
    figures/
      capacity_accuracy.png     # accuracy gain vs hidden dim
      capacity_redundancy.png   # redundancy vs hidden dim
```