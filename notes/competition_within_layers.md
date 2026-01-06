# Competition Within Layers

## The Problem

In a trained neural network, many hyperplanes learn the same cut. Multiple units in a hidden layer converge to nearly identical features. This redundancy is why networks can be pruned aggressively—often 90% of parameters—with minimal performance loss.

This is not a bug in optimization. It is a consequence of architecture. Nothing in a ReLU layer forces units to differentiate.

## Why Output Layers Specialize

At the output layer with cross-entropy loss, softmax normalization creates competition:

```
rⱼ = exp(zⱼ) / Σₖ exp(zₖ)
```

Responsibilities sum to one. If class A takes probability mass, class B loses it. The gradient to each logit is:

```
∂L/∂zⱼ = rⱼ − 𝟙[j = y]
```

Classes that "steal" probability from the correct class receive gradient pushing them away. Classes compete for responsibility. The output prototypes must specialize to different regions of representation space.

This is the implicit EM structure. The loss geometry forces differentiation.

## Why Hidden Layers Don't Specialize

Consider a hidden layer with ReLU activation:

```
a = relu(Wx + b)
```

Each row of W defines a hyperplane. Each unit activates independently based on which side of its hyperplane the input falls.

The gradient to unit i depends on:
1. Whether the unit is active (ReLU gate)
2. How the unit's activation affects the final loss

Crucially, the gradient to unit i does not depend on what other units in the same layer are doing. The Jacobian of ReLU is diagonal:

```
∂aᵢ/∂zⱼ = 𝟙[zᵢ > 0] if i = j, else 0
```

No cross-unit terms. No competition. If units 3 and 7 both help reduce the loss, both receive gradient. They may learn the same hyperplane. Nothing pushes them apart.

## The Geometry of Redundancy

Visualize a 2D input space with a hidden layer of 10 units. Each unit defines a line (hyperplane in 2D). After training:

- Some lines cluster together, nearly parallel, nearly coincident
- Some lines point in directions irrelevant to the task
- A few lines do the actual work of carving up the space

The network "works" because a few units suffice. The rest are deadweight. But we paid for all of them in parameters, computation, and training time.

With competition, each hyperplane would be forced to claim a different region. Ten hyperplanes would carve the space into meaningfully distinct regions. No redundancy. Full utilization.

## Competition Requires Normalization

The implicit EM result shows: responsibilities arise from exponentiation followed by normalization. The normalization is essential. It's what makes responsibilities sum to one. It's what creates the zero-sum dynamic where one unit's gain is another's loss.

ReLU has no normalization. Units operate on independent scales. All can be large. All can be small. There is no budget to divide.

To get competition within a layer, we need some form of normalization across units. Candidates:

**Softmax:** Full normalization. Outputs sum to one. Strong competition. But: loses magnitude information. Output is on simplex. May be too restrictive.

**L2 normalization:** `a = z / ||z||`. Units compete for share of the norm. But: gradient structure is different from softmax. Not clear this gives responsibility-weighted updates.

**Softmax temperature:** `a = softmax(z/τ)`. Temperature τ controls competition strength. High τ: weak competition, nearly uniform. Low τ: strong competition, winner-take-all.

**Grouped softmax:** Partition units into groups. Softmax within groups. Competition is local, not global. Preserves some combinatorial expressiveness.

## The Magnitude Problem

Softmax discards magnitude. If z = [10, 5, 1], softmax outputs responsibilities that say "first unit wins." But the information that the first unit's pre-activation was 10 (not 100, not 1000) is lost.

This matters because downstream layers might need magnitude, not just identity. "Which prototype?" is less informative than "which prototype, and how close?"

Possible solutions:

**Parallel paths:** Compute both relu(z) and softmax(z). Concatenate or combine. Magnitude preserved in one path, competition in the other.

**Multiplicative combination:** `a = z ⊙ softmax(z)`. Each unit's output is its pre-activation scaled by its responsibility. Winners get their magnitude passed through. Losers are suppressed.

**Log-softmax:** `a = log softmax(z) = z − logsumexp(z)`. Subtracts a shared value (the LSE) from all units. Competition exists via the shared term. Relative magnitudes preserved.

**Residual competition:** `a = z + softmax(z)`. Original signal preserved. Competition signal added. But: not clear this creates competitive gradients.

## Gradient Structure

The goal is not just competition in the forward pass but competitive gradients in the backward pass. What matters is the Jacobian ∂a/∂z.

**ReLU Jacobian:**
```
∂aᵢ/∂zⱼ = δᵢⱼ · 𝟙[zᵢ > 0]
```
Diagonal. No cross-unit terms.

**Softmax Jacobian:**
```
∂aᵢ/∂zⱼ = aᵢ(δᵢⱼ − aⱼ)
```
Off-diagonal terms are −aᵢaⱼ. Negative. Raising zⱼ lowers aᵢ for i ≠ j. Competition.

**Jacobian of z ⊙ softmax(z):**
Let r = softmax(z), a = z ⊙ r.
```
∂aᵢ/∂zᵢ = rᵢ + zᵢrᵢ(1 − rᵢ) = rᵢ(1 + zᵢ(1 − rᵢ))
∂aᵢ/∂zⱼ = −zᵢrᵢrⱼ for i ≠ j
```
Off-diagonal terms are negative (assuming z > 0). Competition exists. Magnitude also influences gradient.

**Jacobian of log-softmax:**
```
∂aᵢ/∂zⱼ = δᵢⱼ − rⱼ
```
Clean structure. Diagonal term is 1 − rⱼ. Off-diagonal is −rⱼ. Competition via the shared −rⱼ term.

## What Competition Buys

If competition works as hoped:

1. **No redundant hyperplanes.** Each unit must claim a distinct region or receive no gradient.

2. **Full parameter utilization.** K units means K meaningfully different features, not 3 useful features and K−3 copies.

3. **Implicit regularization.** Competition is a structural pressure toward diversity. May reduce need for explicit regularization.

4. **Interpretability.** Each unit has a clear responsibility region. "Unit 5 responds to inputs in this part of the space" becomes a meaningful statement.

5. **Efficiency.** Smaller networks could match larger redundant networks. Or same-sized networks could achieve higher performance.

## What Competition Costs

1. **Expressiveness.** ReLU allows 2^K activation patterns (each unit on or off). Softmax allows K patterns (which unit wins). The combinatorial capacity may matter.

2. **Gradient flow.** Competition couples all units. Gradient to one unit depends on all others. This could help (forces differentiation) or hurt (complex dynamics, potential instability).

3. **Collapse risk.** Without volume control, one unit can dominate by growing its weights. Need additional mechanisms to prevent winner-take-all collapse.

4. **Compatibility.** Modern architectures assume ReLU-like activations. BatchNorm, residual connections, initialization schemes—all tuned for ReLU. Competitive activations may require rethinking these.

## Open Questions

1. Does competitive activation empirically reduce hyperplane redundancy? (Testable)

2. Which competitive activation best balances specialization and expressiveness?

3. How does competition interact with depth? Does layer 2 benefit from layer 1's specialization?

4. What volume control is needed to prevent collapse?

5. Can competitive activations be made as computationally cheap as ReLU?