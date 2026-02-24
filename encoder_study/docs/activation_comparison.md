# Activation Sign Comparison Experiment

## Motivation

Standard neural networks use ReLU activations that produce non-negative outputs. This experiment asks: **what happens if we flip the sign?**

Using `-ReLU(x) = -max(0, x)` produces non-positive activations. This is equivalent to:
- **ReLU**: The network learns to *maximize* pre-activation values for active features
- **-ReLU**: The network learns to *minimize* pre-activation values for active features

Does this sign flip affect the structure of learned weights?

## Experimental Setup

We train four model variants:

| Variant | Model | Activation | Output Range |
|---------|-------|------------|--------------|
| 1 | Encoder (LSE+InfoMax) | ReLU | [0, ∞) |
| 2 | Encoder (LSE+InfoMax) | -ReLU | (-∞, 0] |
| 3 | SAE (MSE+L1) | ReLU | [0, ∞) |
| 4 | SAE (MSE+L1) | -ReLU | (-∞, 0] |

All variants use:
- Same random seed (reproducible initialization)
- Same architecture (784 → 64)
- Same training hyperparameters
- Same dataset (MNIST)

## Hypotheses

1. **Weight sign flip**: -ReLU weights may be negated versions of ReLU weights, since minimizing `Wx` is equivalent to maximizing `-Wx`.

2. **Structure preservation**: The *structure* of weights (digit prototypes, center-surround patterns) should be preserved regardless of activation sign.

3. **Sparsity difference**: -ReLU may produce different sparsity patterns since the "default" state is 0 (inactive) for different input regions.

4. **SAE vs Encoder**: The effect of sign flip may differ between SAE (reconstruction-driven) and encoder (InfoMax-driven).

## Metrics

For each variant, we measure:

- **Activation statistics**: Mean, std, fraction positive/negative/zero
- **Sparsity (L0)**: Fraction of features active per input
- **Reconstruction MSE**: For SAE variants only
- **Weight visualization**: 8×8 grid of 28×28 feature images

## Running the Experiment

```bash
cd encoder_study

# Run with default config
python scripts/run_activation_comparison.py

# Custom output directory
python scripts/run_activation_comparison.py --output-dir results/my_comparison
```

## Configuration

See `config/activation_comparison.yaml`:

```yaml
model:
  input_dim: 784
  hidden_dim: 64

sae:
  l1_weight: 0.01

infomax:
  lambda_lse: 1.0
  lambda_var: 1.0
  lambda_tc: 1.0

training:
  epochs: 100
  batch_size: 128
  lr: 0.001

seed: 42
```

## Output

Results are saved to `results/activation_comparison/`:

```
results/activation_comparison/
├── encoder_relu.png        # Encoder + ReLU weights
├── encoder_neg_relu.png    # Encoder + (-ReLU) weights
├── sae_relu.png            # SAE + ReLU weights
├── sae_neg_relu.png        # SAE + (-ReLU) weights
├── comparison_all.png      # 2x2 comparison grid
├── comparison_all.pdf      # PDF version
└── results.json            # Numerical metrics
```

## Analysis Questions

1. Are the -ReLU weights simply negated versions of ReLU weights?
2. Do digit prototype structures appear in all variants?
3. How does sparsity compare between ReLU and -ReLU?
4. Does the reconstruction quality of SAE depend on activation sign?
