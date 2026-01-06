# Volume and Collapse in Activations

## The Collapse Problem in Mixture Models

In Gaussian mixture models, each component has a mean μₖ and covariance Σₖ. The log-likelihood for a point x under component k is:

```
log P(x | k) = -½(x - μₖ)ᵀΣₖ⁻¹(x - μₖ) - ½ log det(Σₖ) + const
```

The first term rewards placing μₖ close to x. The second term—the log-determinant—penalizes small covariance. Without it, a component could collapse: shrink Σₖ toward zero, place infinite density on a single point, achieve unbounded likelihood.

The log-determinant is a volume penalty. It says: you can be precise, but precision costs. A component that claims a tiny region of space pays for that precision. This prevents collapse and forces components to maintain reasonable volume.

## Collapse in Neural Mixture Structure

At the output layer of a classifier, logits act as negative distances. Softmax converts to responsibilities. The class with highest logit takes most responsibility.

What prevents one class from dominating all inputs?

In standard training: the labels. Each input has a target class. The correct class is pulled toward responsibility 1. Incorrect classes are pushed away. Supervision distributes inputs across classes.

But consider unsupervised settings, or competitive activations in hidden layers where there are no labels. What prevents collapse?

Nothing structural. A single unit can grow its weights unboundedly, dominate the softmax for all inputs, and starve other units of gradient. The dominant unit takes responsibility ≈ 1 everywhere. Other units get responsibility ≈ 0 everywhere. They receive no learning signal. They die.

This is winner-take-all collapse. One unit wins everything. The layer's capacity reduces to one effective unit.

## The Mechanism of Collapse

Consider a hidden layer with competitive activation:

```
z = Wx + b
a = softmax(z)
```

Suppose unit j has slightly larger weights than others. For most inputs, zⱼ > zᵢ. So rⱼ > rᵢ. Unit j receives more gradient. It learns faster. Its weights grow. Now zⱼ is even larger. The gap widens.

This is positive feedback. Early advantage → more responsibility → more gradient → larger weights → more advantage.

In the limit: one unit has responsibility ≈ 1 for all inputs. Softmax saturates. Other units have responsibility ≈ 0. Their gradients vanish. They stop learning. The layer has collapsed.

## Why GMMs Don't Collapse (Usually)

The log-determinant term counteracts collapse. As a component's covariance shrinks, log det(Σ) → −∞. The penalty grows unboundedly. At some point, the precision gain is offset by the volume penalty. Equilibrium.

Additionally, EM updates for GMMs have specific structure. The M-step sets:

```
μₖ = Σᵢ rᵢₖxᵢ / Σᵢ rᵢₖ
Σₖ = Σᵢ rᵢₖ(xᵢ - μₖ)(xᵢ - μₖ)ᵀ / Σᵢ rᵢₖ
```

Covariance is estimated from data assigned to the component. If a component claims diverse points, it gets large covariance. If it claims few points, its statistics are unstable. The update structure resists extreme configurations.

Neural networks lack both mechanisms. No volume penalty in the loss. No structured parameter updates—just gradient descent.

## Collapse vs. Redundancy

Collapse and redundancy are opposite failure modes:

**Redundancy:** Many units learn the same thing. All units active, but doing identical computation. Waste of parameters. Network can be pruned.

**Collapse:** One unit dominates everything. Other units inactive. Extreme concentration. Network has effectively one hidden unit.

Both are failures of specialization. Redundancy: no pressure to differentiate. Collapse: pressure to differentiate overwhelmed by winner-take-all dynamics.

Competition addresses redundancy but risks collapse. Without competition, units don't differentiate—redundancy. With competition but no volume control, one unit wins—collapse.

We need competition with volume control.

## Candidate Solutions

### Weight Normalization

Force each unit's weight vector to have unit norm:

```
ŵⱼ = wⱼ / ||wⱼ||
z = Ŵx + b
```

Units cannot grow unboundedly. Competition is over direction, not magnitude. A unit cannot dominate by having larger weights—all weights have the same norm.

**Effect:** Limits one axis of collapse. A unit can still dominate via bias, or via better-aligned direction. But unbounded weight growth is blocked.

**Cost:** Constrains the model. May require careful handling of bias terms. Changes the loss landscape.

### Learned Temperature

```
a = softmax(z / τ)
```

Temperature τ controls competition sharpness. High τ → soft competition, uniform-ish responsibilities. Low τ → hard competition, winner-take-all.

If τ is learned or annealed:
- Start with high τ (weak competition). All units receive gradient. They differentiate.
- Gradually lower τ (stronger competition). Specialization sharpens.

**Effect:** Delays collapse. Early training is gentle. Units have time to differentiate before competition intensifies.

**Cost:** Adds hyperparameter or learned parameter. Annealing schedule is another design choice. Final low τ may still collapse.

### Entropy Regularization

Add penalty for low-entropy responsibility distributions:

```
L_total = L_task - λ H(softmax(z))
```

where H is entropy. H is maximized when responsibilities are uniform. Penalty encourages spreading responsibility.

**Effect:** Directly opposes collapse. Dominant unit is penalized. Uniform responsibilities are rewarded.

**Cost:** Tension with task objective. Task may want sharp assignments. Entropy penalty resists. λ must be tuned. May prevent useful specialization if too strong.

### Volume Penalty on Weights

Analogous to log-determinant for GMMs. Penalize configurations where weight vectors are too similar or too concentrated:

```
L_volume = -λ log det(WᵀW)
```

This penalizes rank deficiency in W. If rows of W are similar, WᵀW has small eigenvalues, determinant is small, penalty is large. Forces rows to be diverse.

**Effect:** Directly prevents weight collapse to single direction. Units must span different directions.

**Cost:** Expensive for large layers. Determinant computation is O(K³) for K units. Gradient of log-det adds complexity. May need approximations.

### Minimum Responsibility Threshold

Hard constraint: no unit's average responsibility can fall below threshold.

```
If E[rⱼ] < ε: boost unit j's weights or penalize.
```

**Effect:** Prevents complete starvation. Every unit maintains some presence.

**Cost:** Ad hoc. Threshold is arbitrary. Boost mechanism needs design. May interfere with natural specialization.

### Dropout on Responsibilities

During training, randomly zero some responsibilities and renormalize:

```
r̂ⱼ = rⱼ · mⱼ / Σₖ rₖmₖ
```

where m is a binary mask.

**Effect:** Prevents any unit from being essential. Even dominant unit is sometimes dropped. Other units must learn to compensate. Distributes learning signal.

**Cost:** Noise in training. May slow convergence. Doesn't directly address weight growth.

### Orthogonality Regularization

Encourage weight vectors to be orthogonal:

```
L_ortho = λ ||WᵀW - I||²
```

Off-diagonal terms of WᵀW measure alignment. Penalty pushes them to zero. Diagonal terms pushed to 1 (combined with weight norm).

**Effect:** Forces units to learn orthogonal directions. Maximum diversity. Prevents both collapse and redundancy directly.

**Cost:** Rigid constraint. Orthogonality may not be optimal for the task. K orthogonal vectors in ℝⁿ requires K ≤ n. Limits layer width.

## What Happens in Practice

Modern deep networks use:
- ReLU (no competition, so no collapse)
- BatchNorm (rescales activations, limits magnitude indirectly)
- Weight decay (prevents unbounded weight growth)
- Dropout (distributes learning)
- Careful initialization (starts units in diverse configurations)

These are not principled volume controls. They are heuristics that happen to prevent collapse. BatchNorm is perhaps closest—it normalizes activation statistics, preventing any unit from dominating in magnitude. But it operates on activations, not responsibilities.

If we introduce competitive activations, we may need to revisit these heuristics. BatchNorm before softmax? After? Weight decay may be insufficient—it penalizes magnitude uniformly, not relative dominance.

## The Geometric Picture

Think of K units as K vectors in input space (the rows of W). Each vector defines a direction. The unit "claims" inputs aligned with its direction.

**Redundancy:** Vectors cluster together. Multiple vectors point the same way. Same inputs claimed by multiple units.

**Collapse:** One vector dominates. Its magnitude or bias is so large that it wins regardless of input direction. Other vectors may as well not exist.

**Healthy specialization:** Vectors spread out. Each points in a distinct direction. Each claims a distinct region of input space.

Volume control means: ensure vectors maintain spread. Don't let them cluster (redundancy). Don't let one dominate (collapse). The log-determinant det(WᵀW) measures the volume of the parallelepiped spanned by the weight vectors. Large volume = spread out. Small volume = clustered or collapsed.

## Open Questions

1. **Which volume control works best with which competitive activation?** Weight normalization with softmax? Entropy penalty with log-softmax? Need experiments.

2. **Can volume control be implicit?** BatchNorm already exists. Can it substitute for explicit volume penalties? Or does competition require something stronger?

3. **Does volume control have a computational cost we're willing to pay?** Log-det is expensive. Orthogonality is rigid. Simpler approximations may be needed.

4. **Is there a unified mechanism?** GMMs have one answer: log-determinant from the likelihood. Is there a natural volume term for neural competitive activations that emerges from some principle, rather than being bolted on?

5. **Temperature annealing vs. fixed volume control.** Start soft and sharpen (temperature)? Or maintain fixed diversity pressure (volume penalty)? Or both?

## Relationship to Implicit EM

The paper notes that neural objectives omit the log-determinant. This is why collapse is a risk. The implicit EM dynamics hold—gradients are responsibility-weighted—but nothing prevents the dynamics from concentrating all responsibility in one component.

Volume control is the missing piece. Implicit EM gives us the update rule. Volume control ensures the update rule leads to healthy specialization rather than collapse.

This suggests: a complete "implicit EM activation" might need two components:
1. Competition (softmax or similar) for responsibility-weighted gradients
2. Volume term (in loss or architecture) for collapse prevention

One without the other is incomplete. Competition without volume control collapses. Volume control without competition doesn't specialize.