# Experiment Design: ImplicitEM Layer Ablation

## Research Question

Does implicit EM theory correctly predict the behavior of intermediate layers in supervised networks?

## Predictions

The theory predicts that intermediate layers face the same volume control problem as the unsupervised case in Paper 2. Supervised gradients flow back from the output but do not provide anti-collapse or anti-redundancy guarantees at intermediate layers. Therefore:

1. LSE alone collapses the intermediate representation.
2. Variance prevents dead components.
3. Decorrelation prevents redundancy.
4. The full ImplicitEM layer produces structured mixture components.

These mirror Paper 2's predictions. If the same pattern appears inside a supervised network, the theory extends to the supervised regime. If not, the theory's scope is more constrained than claimed.

## Base Architecture

All configurations share:

```
x ∈ ℝ^784 (flattened MNIST)
    → Linear(784, 64) → Softplus     [distances]
    → [ablation target]               [calibration + volume control]
    → Linear(64, 10) → LayerNorm(10)  [classification head]
    → CrossEntropy(·, labels)          [supervised loss]
```

64 components. Same hidden dimension as Paper 2 for comparability.

## Ablation Configurations

### Config 1: Baseline MLP

```
d = Softplus(W₁x + b₁)
h = LayerNorm(W₂d + b₂)
loss = CE(h, y)
```

No NegLogSoftmin. No auxiliary loss. Standard two-layer supervised network.

### Config 2: NegLogSoftmin Only

```
d = Softplus(W₁x + b₁)
y = NegLogSoftmin(d)
h = LayerNorm(W₂y + b₂)
loss = CE(h, y_label)
```

Calibration without auxiliary loss. Tests whether the NegLogSoftmin Jacobian alone provides meaningful structure.

### Config 3: NegLogSoftmin + LSE

```
loss = CE(h, y_label) + λ · L_LSE(d)
```

EM dynamics without volume control.

**Prediction:** Intermediate collapse. Same as Paper 2 "LSE only."

### Config 4: NegLogSoftmin + LSE + Variance

```
loss = CE(h, y_label) + λ · (L_LSE(d) + λ_var · L_var(d))
```

EM dynamics with anti-collapse but no anti-redundancy.

**Prediction:** No dead components, but high redundancy.

### Config 5: Full ImplicitEM

```
loss = CE(h, y_label) + λ · (L_LSE(d) + λ_var · L_var(d) + λ_tc · L_tc(d))
```

Full volume control.

**Prediction:** No dead components. Low redundancy. Structured intermediate representation.

### Config 6: Variance + Decorrelation Only (No LSE)

```
loss = CE(h, y_label) + λ · (λ_var · L_var(d) + λ_tc · L_tc(d))
```

Volume control without EM dynamics.

**Prediction:** Alive and decorrelated but lower responsibility entropy. Whitening, not clustering.

## Summary Table

| # | Config | NegLogSoftmin | LSE | Var | Decorr | Predicted Intermediate Outcome |
|---|--------|:---:|:---:|:---:|:---:|------|
| 1 | Baseline MLP | — | — | — | — | Unstructured |
| 2 | Calibration only | ✓ | — | — | — | Partial competition from Jacobian |
| 3 | + LSE | ✓ | ✓ | — | — | Collapse |
| 4 | + LSE + var | ✓ | ✓ | ✓ | — | Alive but redundant |
| 5 | Full ImplicitEM | ✓ | ✓ | ✓ | ✓ | Structured mixture |
| 6 | var + tc only | ✓ | — | ✓ | ✓ | Whitened, not clustered |

## Training Protocol

- **Dataset:** MNIST, 60K train / 10K test
- **Hidden dimension:** 64
- **Batch size:** 128
- **Epochs:** 100
- **Optimizer:** Adam, lr = 0.001
- **Seeds:** 3 per configuration (18 runs total)
- **λ = 1.0** (auxiliary loss weight)
- **λ_var = λ_tc = 1.0** when enabled

Same defaults as Paper 2. No tuning.

## Metrics

### Primary (Intermediate Layer)

Applied to distances d on the full test set. These directly parallel Paper 2 Table 5.

- **Dead units:** Components with Var(dⱼ) < 0.01. Range: 0 to 64.
- **Redundancy:** ||Corr(d) − I||²_F (off-diagonal). Lower is better.
- **Responsibility entropy:** E_x[H(softmin(d(x)))]. Higher means softer competition.

### Secondary

- **Classification accuracy:** MNIST test set. The output metric.
- **Weight visualization:** Rows of W₁ reshaped to 28×28. Visual evidence of mixture structure or lack thereof.

## Primary Deliverable

One ablation table:

| Config | Dead Units | Redundancy | Resp. Entropy | Accuracy |
|--------|-----------|-----------|---------------|----------|
| Baseline MLP | ? | ? | ? | ? |
| Calibration only | ? | ? | ? | ? |
| + LSE | ? | ? | ? | ? |
| + LSE + var | ? | ? | ? | ? |
| Full ImplicitEM | ? | ? | ? | ? |
| var + tc only | ? | ? | ? | ? |

Plus one figure: weight visualizations for all six configs.

The table and figure are the result. Everything else is context.

## Success Criteria

The experiment tests the theory's predictions. It succeeds if the results are clear enough to confirm or disconfirm those predictions.

**Theory confirmed if:** The ablation pattern matches Paper 2. LSE collapses, variance prevents death, decorrelation prevents redundancy. The supervised gradient does not substitute for volume control at the intermediate layer.

**Theory disconfirmed if:** The supervised gradient prevents collapse or redundancy without explicit volume control. The baseline MLP or LSE-only config produces healthy intermediate representations. The volume control predictions do not hold when supervision is present.

**Either outcome is a contribution.** Confirmation extends the theory. Disconfirmation constrains it.