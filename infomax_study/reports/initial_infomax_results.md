# Initial InfoMax Experiment Results

## Summary

We conducted exploratory experiments to test whether InfoMax can train a single layer to produce non-redundant, independent features. The answer is yes—but only with the right objective formulation.

The key finding: **InfoMax alone is insufficient. Weight redundancy regularization is necessary to prevent a bias-based loophole in ReLU networks.**

## Experimental Setup

**Architecture:**
```
Input: x ∈ ℝ⁷⁸⁴ (MNIST)
Linear: z = Wx + b ∈ ℝ¹⁶
Activation: ReLU
Loss: InfoMax
```

**InfoMax Loss:**
```
L = -Σⱼ log(Var(aⱼ)) + λ_tc ||Corr(A) - I||²
```

- Entropy term: maximize variance per output
- TC term: minimize correlation between outputs

**Metrics:**
- Weight Redundancy: ||WᵀW - I||² on normalized rows. Lower = more diverse directions.
- Output Correlation: ||Corr(A) - I||². Lower = more independent outputs.
- TC: Total correlation proxy (same as output correlation).
- Effective Rank: How many independent directions in W. Higher = better.
- Dead Units: Units with near-zero activation. Should be 0.

## Experiment 1: Minibatch vs Full Batch

**Question:** Does batch size affect InfoMax optimization?

**Config:** λ_tc = 1.0, 100 epochs, Adam lr=0.001

| Setup | TC | Redundancy | Time |
|-------|-----|------------|------|
| Minibatch (128) | 4.78 | 2.21 | 77s |
| Full batch (60k), 100 steps | 3.10 | 1.13 | 0.5s |
| Full batch (60k), 46.7k steps | 2.99 | 3.07 | 123s |

**Finding 1:** Full batch optimizes TC more effectively. Minibatch noise prevents consistent correlation reduction.

**Finding 2:** Full batch with many steps increases redundancy. The model keeps optimizing entropy while weight geometry drifts.

## Experiment 2: Effect of λ_tc

**Question:** Does increasing TC penalty improve independence?

**Config:** Full batch, 46.7k epochs

| λ_tc | TC | Redundancy | Output Corr |
|------|-----|------------|-------------|
| 1.0 | 2.99 | 3.07 | 3.18 |
| 100.0 | 0.0004 | 3.63 | 0.09 |

**Finding 3:** High λ_tc achieves near-zero TC. Independence goal met.

**Finding 4:** Redundancy increased with higher λ_tc. The model achieved independent outputs via a different mechanism than diverse weights.

## Experiment 3: The ReLU Loophole

**Observation:** Outputs can be independent while weights are redundant.

**Mechanism:** With ReLU, two identical weight vectors with different biases produce independent outputs. One fires for some inputs, the other fires for different inputs. Same projection direction, different thresholds.

**Evidence:** Initial weight redundancy was 0.28 (Xavier init). After training:

| Epoch | Redundancy | TC |
|-------|------------|-----|
| 1 | 0.287 | 4.52 |
| 10 | 0.286 | 0.25 |
| 50 | 0.329 | 0.04 |
| 100 | 0.535 | 0.005 |

Independence improved while weight diversity worsened. The model exploited the bias loophole.

## Experiment 4: Weight Redundancy Regularization

**Solution:** Add explicit weight diversity penalty.

**Modified Loss:**
```
L = -Σⱼ log(Var(aⱼ)) + λ_tc ||Corr(A) - I||² + λ_wr ||WᵀW - I||²
```

**Config:** Minibatch, 100 epochs, λ_tc = 100, λ_wr = 10

| Setup | TC | Redundancy | Output Corr | Eff Rank |
|-------|-----|------------|-------------|----------|
| λ_tc=1, λ_wr=0 | 4.78 | 2.21 | 3.13 | 15.14 |
| λ_tc=100, λ_wr=0 | 0.28 | 7.72 | 0.14 | 14.51 |
| λ_tc=100, λ_wr=10 | 0.27 | **0.00** | 0.12 | 15.36 |

**Finding 5:** Weight redundancy regularization closes the loophole. The model achieves both independent outputs AND diverse weights.

## Key Findings Summary

1. **InfoMax reduces output correlation** when λ_tc is sufficiently high (~100).

2. **Minibatch training is ineffective for TC reduction** due to noisy correlation estimates.

3. **ReLU provides a loophole:** independent outputs via biases, not diverse weights.

4. **Weight redundancy regularization is necessary** to force diverse weight directions.

5. **The complete objective requires three terms:**
   - Entropy: keep units active
   - TC: make outputs independent
   - WR: make weights diverse

## The Final Objective

```
L = -Σⱼ log(Var(aⱼ) + ε) + λ_tc ||Corr(A) - I||² + λ_wr ||WᵀW - I||²
```

With λ_tc ≈ 100, λ_wr ≈ 10, this achieves:
- Near-zero output correlation (independent features)
- Near-zero weight redundancy (diverse hyperplanes)
- Full effective rank (all units utilized)
- Zero dead units

## Connection to Implicit EM

The original motivation was to find an objective for layer-wise implicit EM. The hypothesis:

> InfoMax is the objective. Softmax provides the mechanism (responsibility-weighted updates). InfoMax tells those updates what to achieve: independent, informative features.

**What we learned:**

InfoMax alone defines the *output* goal (independence). But it doesn't constrain the *mechanism* (weight geometry). The EM interpretation requires prototypes—distinct directions in weight space. InfoMax doesn't guarantee this.

The weight redundancy term enforces prototype diversity directly. It's not emergent from the objective—it must be explicit.

**Revised understanding:**

```
InfoMax = what we want (independent outputs)
Weight Redundancy = how we get it (diverse prototypes)
Softmax/EM = learning dynamics (responsibility-weighted updates)
```

All three pieces are necessary.

## Open Questions

1. **Does activation matter?** We only tested ReLU. Does softmax behave differently? Does the loophole exist for all activations?

2. **Does this scale?** Tested K=16. What happens at K=64, K=256?

3. **Does this transfer?** Are InfoMax features useful for downstream tasks?

4. **Is λ_wr sensitive?** What's the range of good values?

5. **Multi-layer?** Can we stack InfoMax-trained layers?

## Next Steps

1. **Activation sweep:** Test identity, ReLU, softmax, tanh with the fixed objective (λ_tc=100, λ_wr=10).

2. **Scaling sweep:** Test hidden_dim ∈ {16, 32, 64, 128}.

3. **Transfer test:** Freeze InfoMax layer, linear probe on MNIST classification.

4. **Visualization:** Plot learned hyperplanes for 2D synthetic data.

## Conclusion

InfoMax + weight redundancy regularization achieves the goal: independent, non-redundant features from unsupervised training.

The key insight is that **output independence ≠ weight diversity**. Both must be optimized explicitly. The ReLU bias loophole demonstrates that neural networks will find shortcuts if the objective allows them.

This suggests a general principle: if you want a geometric property (diverse hyperplanes), optimize for it directly. Don't assume it will emerge from an indirect objective.