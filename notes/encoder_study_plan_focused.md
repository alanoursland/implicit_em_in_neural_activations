# Encoder Study: Focused Experiment Plan

## Paper Claim

Decoder-free sparse autoencoders arise naturally from implicit EM when log-sum-exp objectives are combined with InfoMax regularization.

## Contribution

The derivation. We derived a known-useful architecture from first principles. The path is the novelty, not the destination.

## What We Need to Prove

| Claim | Evidence |
|-------|----------|
| Gradient = Responsibility | One scatter plot |
| Collapse without InfoMax | One ablation |
| It works | One benchmark |

## Experiment 1: Verify the Theorem

**Claim:** ∂L_LSE/∂E_j = r_j

**Method:**
- Single forward/backward pass
- Record computed responsibility r_j = softmax(-E_j)
- Record actual gradient ∂L_LSE/∂E_j

**Output:** Scatter plot. Should be identity line.

**Time:** 30 minutes.

## Experiment 2: Ablation

**Claim:** InfoMax prevents collapse.

**Method:** Train four configurations:

| Config | LSE | Variance | Decorrelation |
|--------|-----|----------|---------------|
| A | ✓ | | |
| B | ✓ | ✓ | |
| C | ✓ | ✓ | ✓ |
| D | | ✓ | ✓ |

**Metrics:**
- Dead units (variance < ε)
- Redundancy (||Corr - I||²)
- Feature usage distribution

**Output:** Table + one figure showing feature similarity matrices.

**Expected:**
- Config A: collapse (one component wins)
- Config B: no dead units, some redundancy
- Config C: stable, diverse
- Config D: no EM structure, different behavior

**Time:** 2 hours.

## Experiment 3: Benchmark

**Claim:** It works as well as standard SAEs.

**Dataset:** GPT-2 small MLP activations or MNIST.

**Baseline:** Standard SAE (encoder + decoder + L1 sparsity).

**Metrics:**
- Reconstruction MSE (using W^T as implicit decoder)
- Sparsity (L0)
- Parameter count

**Output:** Table comparing metrics.

**Expected:** Comparable quality, fewer parameters.

**Time:** 4 hours.

## Paper Structure

1. **Introduction** (1 page)
2. **Background** (1 page): LSE identity, collapse in mixtures
3. **Derivation** (2 pages): LSE + InfoMax
4. **Experiments** (2 pages): Three figures
5. **Discussion** (1 page)

## Three Figures

1. **Figure 1:** Gradient = Responsibility scatter plot
2. **Figure 2:** Ablation grid (similarity matrices or bar chart)
3. **Figure 3:** Benchmark comparison table/chart

## What We Don't Need

- Prove it's better than everything
- Exhaustive hyperparameter sweeps
- Multiple datasets
- Novel interpretability findings
- State-of-the-art claims

## Timeline

| Task | Time |
|------|------|
| Implement LSE loss | 1 hour |
| Experiment 1 (theorem) | 30 min |
| Experiment 2 (ablation) | 2 hours |
| Experiment 3 (benchmark) | 4 hours |
| Figures | 2 hours |
| Writing | 8 hours |
| **Total** | ~2 days |

## Success Criteria

- Figure 1 shows identity line (theorem holds)
- Figure 2 shows collapse without InfoMax (motivation holds)
- Figure 3 shows competitive performance (utility holds)

If all three hold, the paper validates the derivation. The theory does the heavy lifting.