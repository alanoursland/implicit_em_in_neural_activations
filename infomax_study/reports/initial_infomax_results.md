# Initial InfoMax Experiment Results

## Summary

We conducted exploratory experiments to test whether InfoMax can train a single layer to produce non-redundant, independent features. The key findings:

1. **InfoMax alone is insufficient.** Without EM structure, ReLU networks exploit a bias loophole to achieve independent outputs without diverse weights.

2. **Softmax provides implicit weight regularization.** Adding softmax after the activation creates EM structure that prevents the loophole and stabilizes training.

3. **EM structure improves optimization.** With softmax, SGD and Adam perform similarly. Without softmax, Adam significantly outperforms SGD.

4. **The mechanism matters, not just the objective.** The same InfoMax objective produces different geometry depending on whether EM structure is present.

## Experimental Setup

**Architecture A (No EM structure):**
```
Input → Linear → ReLU → InfoMax Loss
```

**Architecture B (With EM structure):**
```
Input → Linear → ReLU → Softmax → InfoMax Loss
```

**InfoMax Loss:**
```
L = -Σⱼ log(Var(aⱼ) + ε) + λ_tc ||Corr(A) - I||² + λ_wr ||WᵀW - I||²
```

- Entropy term: maximize variance per output (keeps units active)
- TC term: minimize correlation between outputs (independence)
- WR term: minimize weight redundancy (diverse hyperplanes)

**Metrics:**
- Weight Redundancy: ||WᵀW - I||² on normalized rows. Lower = more diverse directions.
- Output Correlation: ||Corr(A) - I||². Lower = more independent outputs.
- Effective Rank: Dimensionality of weight space. Higher = better utilization.
- Dead Units: Units with near-zero activation. Should be 0.

## Part 1: Architecture Without EM Structure

### The Bias Loophole

**Setup:** Linear → ReLU → InfoMax, full batch training, λ_tc = 1.0, λ_wr = 0.0

| Epoch | Redundancy | Output Corr | TC |
|-------|------------|-------------|-----|
| 1 | 0.287 | 2.98 | 4.52 |
| 10 | 0.286 | 0.25 | 0.25 |
| 50 | 0.329 | 0.08 | 0.04 |
| 100 | 0.535 | 0.04 | 0.005 |

**Finding:** Output independence improved while weight diversity worsened.

The model achieved independent outputs by exploiting ReLU's bias mechanism: two identical weight vectors with different biases fire on different inputs. Same direction, different thresholds. The outputs are independent, but the weights are redundant.

### Increasing λ_tc Makes It Worse

**Setup:** Linear → ReLU → InfoMax, minibatch training, λ_wr = 0.0

| λ_tc | TC | Redundancy | Output Corr |
|------|-----|------------|-------------|
| 1.0 | 4.78 | 2.21 | 3.13 |
| 100.0 | 0.28 | 7.72 | 0.14 |

Higher TC pressure achieved better independence but worse redundancy. The model exploited the loophole harder.

### Explicit Weight Regularization Closes the Loophole

**Setup:** Linear → ReLU → InfoMax, minibatch training

| λ_tc | λ_wr | TC | Redundancy | Output Corr |
|------|------|-----|------------|-------------|
| 1.0 | 0.0 | 4.78 | 2.21 | 3.13 |
| 100.0 | 0.0 | 0.28 | 7.72 | 0.14 |
| 100.0 | 10.0 | 0.27 | **0.00** | 0.12 |

Adding explicit weight redundancy penalty achieved both goals: independent outputs AND diverse weights.

**Conclusion for Architecture A:** Without EM structure, InfoMax requires explicit weight regularization to prevent the bias loophole.

## Part 2: Architecture With EM Structure

### Softmax Provides Implicit Regularization

**Setup:** Linear → ReLU → Softmax → InfoMax, minibatch training, λ_tc = 0.0, λ_wr = 0.0

| Epochs | Redundancy | Output Corr | Loss |
|--------|------------|-------------|------|
| 100 | 2.85 | 1.07 | 46.53 |
| 200 | 2.89 | 1.08 | 47.60 |

**Finding:** Redundancy stabilizes without any explicit regularization.

Compare to Architecture A without regularization:
- ReLU alone (100k epochs): Redundancy drifted to 3.52 and kept increasing
- ReLU → Softmax (200 epochs): Redundancy stable at 2.89

**Why softmax prevents the loophole:**

Softmax outputs sum to 1. If two units have similar weights, they produce similar pre-softmax values. Softmax forces them to compete—the gradient pushes them apart. The loophole (independent outputs from redundant weights) is structurally impossible.

### TC Pressure is Counterproductive

**Setup:** Linear → ReLU → Softmax → InfoMax, varying λ_tc

| λ_tc | Redundancy | Output Corr |
|------|------------|-------------|
| 0.0 | 2.85 | 1.07 |
| 1.0 | 3.31 | 1.08 |
| 10.0 | 1.80 | 13.03 |

**Finding:** High λ_tc fights a losing battle.

Softmax outputs are constrained to the simplex—they cannot be truly independent. Pushing λ_tc high distorts the optimization without achieving the impossible goal.

With EM structure, the entropy term alone is sufficient. Competition provides implicit regularization.

### SGD ≈ Adam With EM Structure

**Architecture A (no softmax):**
- SGD at lr=0.001: Very slow, poor results
- SGD at lr=10.0: Required for reasonable convergence
- Adam at lr=0.001: Works well

**Architecture B (with softmax):**
- SGD at lr=0.001: Works
- Adam at lr=0.001: Similar performance

| Architecture | Optimizer | lr | Final Loss | Redundancy |
|--------------|-----------|-----|------------|------------|
| No Softmax | SGD | 0.001 | (very slow) | - |
| No Softmax | Adam | 0.001 | -202.07 | 2.21 |
| Softmax | SGD | 0.001 | 47.98 | 2.86 |
| Softmax | Adam | 0.001 | 46.53 | 2.85 |

**Finding:** EM structure normalizes the optimization landscape.

Classical EM has no learning rate—responsibilities determine the "right size" step automatically. Softmax approximates this by providing responsibility-weighted gradients. Adam's adaptive scaling becomes unnecessary.

## Key Findings

### 1. The Bias Loophole

ReLU networks can achieve independent outputs without diverse weights by using bias diversity. Two identical hyperplanes with different offsets fire on different inputs.

InfoMax optimizes for output independence. It doesn't care about weight geometry. The loophole is a valid solution to the objective.

### 2. EM Structure Closes the Loophole

Softmax creates competition. Units must differentiate to receive gradient. Redundant weights produce nearly identical pre-softmax values, losing the competition.

The loophole (independent outputs from redundant weights) is structurally impossible after softmax—outputs sum to 1.

### 3. Explicit Regularization vs. Implicit Regularization

| Architecture | Required for non-redundant weights |
|--------------|-----------------------------------|
| Linear → ReLU → InfoMax | Explicit λ_wr regularization |
| Linear → ReLU → Softmax → InfoMax | Nothing (implicit via competition) |

### 4. Optimization Landscape

EM structure (softmax) creates a cleaner gradient landscape:
- SGD works at normal learning rates
- Adam provides no significant advantage
- No need for adaptive optimization tricks

Without EM structure:
- SGD requires very high learning rates
- Adam significantly outperforms SGD
- Optimization is less stable

## Theoretical Interpretation

### What InfoMax Provides

InfoMax is the **objective**: maximize information, minimize redundancy in outputs.

It specifies *what* we want but not *how* to get it.

### What EM Structure Provides

Softmax creates **responsibilities**. The backward pass distributes gradients according to responsibility. This is the *mechanism*.

The mechanism constrains the optimization:
- Units compete for responsibility
- Competition forces differentiation
- Differentiation requires diverse weights

### The Complete Picture

```
InfoMax = what we want (independent, informative outputs)
EM (Softmax) = how we get it (competition via responsibilities)
```

Both pieces are necessary:
- InfoMax without EM: achieves objective via loopholes
- EM without InfoMax: no objective to optimize
- InfoMax with EM: achieves objective via intended mechanism

## Comparison of Architectures

| Property | Linear→ReLU→InfoMax | Linear→ReLU→Softmax→InfoMax |
|----------|--------------------|-----------------------------|
| EM structure | No | Yes |
| Bias loophole | Yes | No |
| Requires λ_wr | Yes | No |
| Requires λ_tc | Yes (high) | No (counterproductive) |
| SGD works | Poorly | Yes |
| Adam needed | Yes | No |
| Weight redundancy | Drifts | Stable |
| Output independence | Achievable | Constrained by simplex |

## Practical Recommendations

1. **Use EM structure (softmax)** if you want implicit weight regularization.

2. **Don't push λ_tc** with softmax outputs—they can't be truly independent.

3. **λ_wr is only needed** without EM structure.

4. **SGD is fine** with EM structure. Adam provides no significant benefit.

5. **The entropy term is primary.** It keeps all units active. Competition handles differentiation.

## Open Questions

1. **Does the pre-softmax activation matter?** Identity vs ReLU vs Tanh before softmax.

2. **Does this scale?** K=16 tested. What about K=64, K=256?

3. **Do these features transfer?** Linear probe on downstream tasks.

4. **Multi-layer stacking?** Can we compose InfoMax-trained layers?

5. **Other competition mechanisms?** Sparsemax, GroupSoftmax, etc.

## Next Experiment

**Sweep:** Effect of pre-softmax activation

```yaml
activation: [identity, relu, tanh]
hidden_dim: [16, 32, 64]
lambda_tc: [0.0]
lambda_wr: [0.0]
seed: [1, 2, 3, 4, 5]
```

Architecture: Linear → Activation → Softmax → InfoMax

**Question:** Does the activation before softmax affect weight redundancy and feature quality?

## Conclusion

The main finding is that **the mechanism matters, not just the objective**.

InfoMax specifies what we want. EM structure (softmax) provides the mechanism to achieve it without loopholes. Together, they produce non-redundant, informative features with stable optimization.

This validates the hypothesis from the theoretical notes: implicit EM requires both an objective (InfoMax) and a mechanism (softmax/responsibilities). The mechanism constrains the solution space to geometrically meaningful optima.