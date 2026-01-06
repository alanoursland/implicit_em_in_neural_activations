# Experiment: InfoMax Across Activations

## Objective

Test whether InfoMax produces non-redundant features, and how activation choice affects the result.

## Background

The implicit EM framework establishes that gradient descent performs EM at the output layer via responsibility-weighted updates. We hypothesize that InfoMax is the right objective for unsupervised learning in this framework—it maximizes information while preventing redundancy and collapse.

This experiment tests the hypothesis on a single layer, varying only the activation function.

## Architecture

```
Input: x ∈ ℝⁿ
Linear: z = Wx + b ∈ ℝᴷ
Activation: a = f(z) ∈ ℝᴷ
Loss: InfoMax(a)
```

No decoder. No reconstruction. Pure InfoMax on the output.

## InfoMax Loss

```
L = -Σⱼ H(aⱼ) + λ TC(a)
```

First term: maximize marginal entropy of each output.
Second term: minimize total correlation (dependence between outputs).

**Approximation via batch statistics:**

For a batch of N samples producing activations A (N × K matrix):

Marginal entropy term (Gaussian approximation):
```
H(aⱼ) ≈ ½ log(2πe · Var(aⱼ))
```

Total correlation term (decorrelation proxy):
```
TC ≈ ||Corr(A) - I||²
```

Where Corr(A) is the K × K correlation matrix of columns of A.

Combined loss:
```
L = -Σⱼ log(Var(aⱼ) + ε) + λ ||Corr(A) - I||²
```

This is differentiable and cheap to compute.

## Activations to Test

| Activation | Formula | Properties |
|------------|---------|------------|
| Identity | a = z | Unbounded, linear |
| ReLU | a = max(0, z) | Sparse, non-negative, dead units possible |
| Softmax | a = exp(z) / Σexp(z) | Normalized, sum-to-one, competitive |
| Tanh | a = tanh(z) | Bounded (-1, 1), smooth |
| Leaky ReLU | a = max(0.01z, z) | Sparse-ish, no dead units |
| Softplus | a = log(1 + exp(z)) | Smooth ReLU, non-negative |

## Dataset

**Primary: MNIST**
- n = 784 input dimensions
- Well understood
- Enough structure to learn meaningful features

**Secondary: 2D synthetic**
- n = 2 input dimensions
- Allows visualization of weight vectors as lines
- Gaussian clusters or uniform

## Hyperparameters

- Hidden units K: 16, 32, 64
- Batch size: 128
- Optimizer: Adam
- Learning rate: 1e-3
- λ (TC weight): 1.0 (sweep 0.1, 1.0, 10.0)
- Training epochs: 100
- Seeds: 10 per configuration

## Metrics

### Redundancy

**Weight correlation:**
```
R_W = ||WᵀW / (||W||² ) - I/K||²
```
Measures whether weight vectors point in similar directions. Lower is better.

**Effective rank:**
```
σ = singular values of W
eff_rank = (Σσᵢ)² / Σσᵢ²
```
Ranges from 1 (all weights identical) to min(n, K) (full rank). Higher is better.

### Dead Units

**Activation rate:**
For each unit j, fraction of samples where |aⱼ| > ε.
Dead unit if activation rate < 0.01.

Count of dead units. Lower is better.

### Collapse

**Responsibility concentration (for softmax):**
```
max_j E[rⱼ]
```
If close to 1, one unit dominates. Should be close to 1/K for balanced usage.

**Variance ratio:**
```
max_j Var(aⱼ) / min_j Var(aⱼ)
```
High ratio indicates some units much more active than others.

### Independence Achieved

**Output correlation:**
```
||Corr(A) - I||²
```
The objective itself. Lower is better.

**Total correlation (if estimated):**
```
TC(a) = Σⱼ H(aⱼ) - H(a)
```
Zero if independent. Lower is better.

### Optimization

**Final loss:** Mean and std across seeds.

**Convergence speed:** Epochs to reach 90% of final loss.

**Loss variance:** Std of final loss across seeds. Lower indicates more stable optimization.

## Procedure

For each activation in {identity, ReLU, softmax, tanh, leaky ReLU, softplus}:

For each K in {16, 32, 64}:

For each seed in {1, ..., 10}:

1. Initialize W with Xavier initialization
2. Initialize b = 0
3. Train for 100 epochs with InfoMax loss
4. Record all metrics at end of training
5. Save weight matrix W

## Analysis

### Primary Question

Does InfoMax produce non-redundant features?

Compare effective rank and weight correlation to:
- Random initialization (untrained)
- Autoencoder baseline (reconstruction loss)

If InfoMax yields higher effective rank and lower weight correlation, hypothesis confirmed.

### Secondary Question

How does activation choice affect results?

Compare across activations:
- Which has lowest redundancy?
- Which has most dead units?
- Which has best independence?
- Which has most stable optimization?

Look for tradeoffs. Maybe ReLU has low redundancy among survivors but many dead units. Maybe softmax has no dead units but collapses.

### Tertiary Question

Does λ matter?

Sweep λ ∈ {0.1, 1.0, 10.0}.

- Low λ: weak independence pressure, might allow redundancy
- High λ: strong independence pressure, might hurt optimization

Find the sweet spot.

## Expected Outcomes

**Optimistic:**
- InfoMax yields high effective rank (non-redundant)
- One activation clearly wins
- Results are stable across seeds

**Pessimistic:**
- InfoMax doesn't improve over random init
- All activations collapse or have dead units
- High variance across seeds

**Likely:**
- InfoMax helps but doesn't eliminate redundancy
- Activations show different failure modes
- λ tuning matters

## Baselines

### Random (untrained)
Xavier-initialized W, no training. Measures: what does optimization contribute?

### Reconstruction
```
L = ||x - W'a||²
```
Standard autoencoder. Measures: what does InfoMax contribute beyond just "use all units"?

### Variance only
```
L = -Σⱼ log(Var(aⱼ) + ε)
```
No independence term. Just maximize variance. Measures: what does the TC term contribute?

## Deliverables

1. Table: metrics by activation and K
2. Plot: effective rank vs activation
3. Plot: dead units vs activation
4. Plot: training curves by activation
5. Visualization: learned weight vectors for 2D case
6. Statistical tests: is InfoMax significantly better than baselines?

## Follow-up Experiments

If results are promising:

1. **Depth:** Stack two layers. Does InfoMax compose?
2. **Transfer:** Freeze and linear probe. Do InfoMax features transfer?
3. **Scale:** Larger K, larger data. Does it hold?
4. **Downstream:** Does low redundancy predict good task performance?