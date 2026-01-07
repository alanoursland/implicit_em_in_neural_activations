# Experiment 3: Benchmark Comparison

## Objective

Compare the decoder-free encoder against a standard sparse autoencoder to validate that the principled objective produces useful features in practice.

**Hypothesis:** The decoder-free model should achieve comparable feature quality with fewer parameters, since the decoder is theoretically redundant (Section 6.3).

## Method

### Models

**Decoder-Free Encoder (Ours):**
- Architecture: Linear (784 → 64) + ReLU
- Loss: LSE + Variance + Decorrelation (Equation 7)
- Parameters: 50,240

**Standard SAE:**
- Architecture: Linear (784 → 64) + ReLU + Linear (64 → 784)
- Loss: MSE reconstruction + L1 sparsity (λ = 0.01)
- Parameters: 101,200

### Evaluation

Both models produce 64-dimensional ReLU features. We evaluate on:

1. **Linear Probe Accuracy:** Train logistic regression on frozen features, report MNIST test accuracy. Measures feature quality for downstream tasks.

2. **L0 Sparsity:** Mean number of features with activation > 0.01 per input. Lower = sparser.

3. **Parameter Count:** Total trainable parameters.

4. **Normalized Reconstruction MSE:** Reconstruction error with scale-normalized activations. Note: SAE uses its trained decoder; ours uses W^T (untrained).

### Training

- Dataset: MNIST (60K train, 10K test)
- Epochs: 100
- Batch size: 128
- Optimizer: Adam (lr = 0.001)
- Seeds: 5 per model

## Results

| Metric | Decoder-Free (Ours) | Standard SAE |
|--------|---------------------|--------------|
| Linear Probe Accuracy | **93.43% ± 0.38%** | 90.26% ± 0.32% |
| L0 Sparsity | **17.2 / 64 (26.8%)** | 32.2 / 64 (50.3%) |
| Parameters | **50,240** | 101,200 |
| Normalized Recon MSE | 0.143 ± 0.001 | **0.026 ± 0.001** |

## Interpretation

### Feature Quality: Ours is Better (+3.2%)

The decoder-free encoder achieves 93.4% linear probe accuracy versus 90.3% for the standard SAE—a 3.2 percentage point improvement. This gap is consistent across all seeds (range: 92.98–93.86% vs 89.65–90.61%).

For context, logistic regression on raw pixels achieves ~92%. Our 64-dimensional features outperform 784 raw pixels while the SAE's features underperform them. The decoder-free objective produces more linearly separable representations.

### Sparsity: Ours is Sparser (2×)

Despite having no explicit sparsity penalty, the decoder-free encoder activates only 27% of features per input, versus 50% for the SAE with L1 regularization.

This is surprising. The SAE explicitly penalizes L1 norm; we do not. Yet our features are twice as sparse. The decorrelation penalty (Equation 5) may induce sparsity indirectly: to be uncorrelated, features must activate on different inputs, which naturally limits co-activation.

### Parameters: 50% Reduction

The decoder-free model uses 50,240 parameters versus 101,200—a 50.4% reduction. This is the expected consequence of removing the decoder.

### Reconstruction: SAE Wins (5×)

The standard SAE achieves 5× lower reconstruction error (0.026 vs 0.143). This is expected: the SAE is explicitly trained to minimize reconstruction error with a learned decoder. Our model uses the transposed encoder weights W^T, which are never trained for reconstruction.

This comparison is not entirely fair—SAE uses a trained decoder, we use an untrained one. A fairer comparison would use W^T for both models. However, reconstruction is not our objective. The linear probe results demonstrate that information is preserved; it is simply encoded differently.

## Summary

| Claim | Result |
|-------|--------|
| Comparable accuracy | **Better** (+3.2%) |
| Comparable sparsity | **Better** (2× sparser) |
| Fewer parameters | **Confirmed** (50% reduction) |
| Worse reconstruction | Confirmed (5× higher MSE) |

The decoder-free objective exceeds expectations. We predicted competitive performance; we achieved superior performance on feature quality metrics while using half the parameters.

## Limitations

- MNIST only; results may differ on more complex datasets
- Single hidden dimension (64); scaling behavior unknown
- L1 weight (0.01) not tuned; different values might improve SAE
- Reconstruction comparison is unfair (trained vs untrained decoder)

## Conclusion

The benchmark validates the paper's central claim: the decoder is unnecessary. The decoder-free objective produces features that are more useful (higher probe accuracy), sparser (lower L0), and cheaper (fewer parameters) than standard SAE features. The only cost is reconstruction fidelity—which was never the goal.

## Code

```bash
python scripts/run_benchmark.py --config config/benchmark.yaml --output-dir results/benchmark
```

## Files

- Results: `results/benchmark/benchmark_results.json`
- Figures: `figures/fig3_benchmark.pdf` (to be generated)