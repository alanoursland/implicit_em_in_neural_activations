# Experiment 1: Theorem Verification

## Objective

Verify that the gradient of the log-sum-exp loss with respect to each component energy equals its responsibility exactly, as stated in Equation 2:

$$\frac{\partial L_{\text{LSE}}}{\partial E_j} = r_j = \frac{\exp(-E_j)}{\sum_k \exp(-E_k)}$$

This identity is the foundation of implicit EM: it shows that backpropagation through an LSE objective automatically computes responsibilities without explicit E-step computation.

## Method

1. Create random activations: `a = torch.randn(64, 128)`
2. Compute LSE loss: $L = -\sum_i \log \sum_j \exp(-a_{ij})$
3. Compute responsibilities: $r = \text{softmax}(-a)$
4. Backpropagate to obtain gradients: `loss.backward()`
5. Compare `a.grad` to `r`

No training is involved. This is a single forward-backward pass verifying an algebraic identity.

## Results

| Metric | Value |
|--------|-------|
| Correlation | 1.0000 |
| Max absolute error | 4.47e-08 |
| Mean absolute error | 2.79e-09 |

The errors are at floating-point precision. The gradient equals the responsibility exactly.

## Figure

![Theorem Verification](../figures/fig1_theorem.png)

The scatter plot shows gradient (y-axis) versus responsibility (x-axis) for all 8,192 values (64 samples × 128 components). All points lie exactly on the y = x identity line.

## Interpretation

This result confirms that the LSE loss has the exact property claimed in Section 2.1:

- The gradient with respect to each energy IS its responsibility
- No approximation is involved; this is an algebraic identity
- Backpropagation through LSE performs the E-step implicitly

This validates the theoretical foundation of the decoder-free sparse autoencoder. The implicit EM machinery (Oursland, 2025) is correctly implemented.

## Code

```bash
python scripts/verify_theorem.py --output figures/fig1_theorem.pdf
```

## Conclusion

**Theorem verified.** The gradient of the LSE loss equals the responsibility to within floating-point precision. Experiment 1 is complete.