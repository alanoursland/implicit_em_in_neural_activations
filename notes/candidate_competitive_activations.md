# Candidate Competitive Activations

## The Goal

We want an activation function f such that when a layer computes a = f(Wx + b), the backward pass distributes gradients competitively among units. Units that "claim" an input should receive most of the gradient. Units that don't claim it should receive little.

The gradient structure we want resembles softmax: cross-unit terms in the Jacobian that create zero-sum dynamics. But we may also want properties softmax lacks: preserved magnitude, computational efficiency, compatibility with deep stacking.

## Baseline: ReLU

```
a = relu(z) = max(0, z)
```

**Forward:** Each unit outputs its pre-activation if positive, else zero.

**Jacobian:**
```
∂aᵢ/∂zⱼ = δᵢⱼ · 𝟙[zᵢ > 0]
```

Diagonal. No cross-unit terms. Units are independent.

**Gradient flow:** Each unit receives gradient scaled by whether it was active. Dead units (always inactive) receive no gradient. But active units don't compete—all active units receive their full gradient signal.

**Competition:** None.

**Expressiveness:** High. K units can create up to 2^K linear regions (activation patterns). Combinatorial capacity.

**Magnitude:** Preserved for positive values. Lost (zeroed) for negative values.

ReLU is the baseline against which competitive activations must be measured. Any replacement must offer benefits that justify departing from ReLU's simplicity and expressiveness.

---

## Candidate 1: Softmax

```
a = softmax(z), where aᵢ = exp(zᵢ) / Σₖ exp(zₖ)
```

**Forward:** Outputs a probability distribution over units. Sum to one. All positive.

**Jacobian:**
```
∂aᵢ/∂zⱼ = aᵢ(δᵢⱼ − aⱼ)
```

Off-diagonal terms: −aᵢaⱼ < 0. Raising zⱼ lowers aᵢ for i ≠ j.

**Gradient flow:** If unit j dominates (aⱼ ≈ 1), it receives most gradient. Other units receive gradient proportional to their (small) responsibility, pushing them to differentiate.

**Competition:** Strong. Zero-sum by construction.

**Expressiveness:** Low. Output is on the K−1 simplex. Effectively encodes "which unit wins" with soft interpolation. K effective patterns, not 2^K.

**Magnitude:** Lost entirely. Only relative ranking matters. z = [10, 5, 1] and z = [100, 50, 10] produce the same output.

**Assessment:** Maximum competition, minimum expressiveness. Too restrictive for hidden layers. The simplex constraint impoverishes the representation.

---

## Candidate 2: Log-Softmax

```
a = log_softmax(z) = z − logsumexp(z)
```

Equivalently: aᵢ = zᵢ − log Σₖ exp(zₖ)

**Forward:** Subtracts a shared value (the LSE) from all units. Output can be any real number. Not normalized to simplex.

**Jacobian:**
```
∂aᵢ/∂zⱼ = δᵢⱼ − softmax(z)ⱼ = δᵢⱼ − rⱼ
```

Diagonal terms: 1 − rⱼ. Off-diagonal terms: −rⱼ.

**Gradient flow:** The −rⱼ term appears in all rows. The dominant unit (high rⱼ) suppresses gradient to all units including itself. Competition exists but is mediated through subtraction rather than normalization.

**Competition:** Moderate. All units are pulled toward the mean log-probability. Dominant units pull others down; weak units pull others up.

**Expressiveness:** Higher than softmax. Output is not on simplex. Relative magnitudes preserved in the sense that aᵢ − aⱼ = zᵢ − zⱼ. Differences unchanged.

**Magnitude:** Partially preserved. Absolute scale lost (shifted by LSE), but relative structure intact.

**Assessment:** A middle ground. Competition via shared subtraction. More expressive than softmax. But the constant shift may interact strangely with downstream layers. Resembles a residual connection in structure.

---

## Candidate 3: Softmax-Weighted Pre-activation

```
a = z ⊙ softmax(z)
```

Element-wise product of pre-activation and responsibility.

**Forward:** Each unit outputs its pre-activation scaled by its responsibility. Winners (high z, high r) get large output. Losers (low r) are suppressed regardless of z magnitude.

**Jacobian:**
Let r = softmax(z).
```
∂aᵢ/∂zᵢ = rᵢ + zᵢ · rᵢ(1 − rᵢ) = rᵢ(1 + zᵢ(1 − rᵢ))
∂aᵢ/∂zⱼ = zᵢ · rᵢ · (−rⱼ) = −zᵢrᵢrⱼ for i ≠ j
```

**Gradient flow:** Off-diagonal terms are −zᵢrᵢrⱼ. Sign depends on zᵢ. For positive pre-activations, competition exists. Magnitude of pre-activation amplifies competitive effect.

**Competition:** Present but conditional. Strength depends on magnitudes. Zero pre-activation means no competition from that unit.

**Expressiveness:** Moderate. Not constrained to simplex. Magnitudes partially preserved—scaled by responsibility, not discarded.

**Magnitude:** Transformed. Large z with large r → large output. Large z with small r → suppressed. The scaling is multiplicative.

**Assessment:** Interesting hybrid. Competition exists. Magnitude influences result. But the multiplicative interaction may cause instability—large z amplifies both signal and competitive pressure.

---

## Candidate 4: Softmax Plus Pre-activation

```
a = z + softmax(z)
```

Additive combination.

**Forward:** Original pre-activation plus responsibility. Output unbounded.

**Jacobian:**
```
∂aᵢ/∂zⱼ = δᵢⱼ + rᵢ(δᵢⱼ − rⱼ)
        = δᵢⱼ(1 + rᵢ(1 − rᵢ)) − rᵢrⱼ(1 − δᵢⱼ)
```

Diagonal: 1 + rᵢ(1 − rᵢ) > 1. Off-diagonal: −rᵢrⱼ < 0.

**Gradient flow:** Diagonal terms boosted. Off-diagonal terms negative. Competition exists via the softmax component.

**Competition:** Weak. The identity component (∂z/∂z = I) dominates. Softmax competition is additive, not multiplicative.

**Expressiveness:** High. The z term preserves full pre-activation information. Softmax term adds soft assignment signal.

**Magnitude:** Fully preserved in the z component.

**Assessment:** Minimal modification to ReLU-like behavior. Competition is present but may be too weak to force specialization. The softmax term is a perturbation, not a restructuring.

---

## Candidate 5: Grouped Softmax

```
Partition units into G groups of size K/G.
Within each group: a_g = softmax(z_g)
Across groups: independent.
```

**Forward:** Competition within groups. Independence across groups.

**Jacobian:** Block diagonal. Each block is a softmax Jacobian. No cross-group terms.

**Gradient flow:** Units compete with their group-mates, not with all units.

**Competition:** Local. Strong within groups. None across groups.

**Expressiveness:** Intermediate. G groups with K/G options each gives (K/G)^G effective patterns. More than pure softmax (K), less than ReLU (2^K).

**Magnitude:** Lost within groups (softmax). Group outputs could be concatenated.

**Assessment:** Tunable competition/expressiveness tradeoff via group size. Group size 2 gives pairwise competition. Group size K gives full softmax. May be useful for controlled experiments.

---

## Candidate 6: Sparsemax

```
a = sparsemax(z) = argmin_{p ∈ Δ} ||p − z||²
```

Euclidean projection onto the simplex. Unlike softmax, produces exact zeros.

**Forward:** Sparse probability distribution. Some units get exactly zero weight. Others share the remaining mass.

**Jacobian:** Piecewise linear. Non-zero only for the "support" (active units). Within support, resembles softmax Jacobian but renormalized.

**Gradient flow:** Inactive units receive exactly zero gradient. Active units compete among themselves.

**Competition:** Strong among active units. Inactive units are hard-zeroed like ReLU.

**Expressiveness:** Between softmax and ReLU. Sparse like ReLU. Normalized like softmax.

**Magnitude:** Lost. Output is on simplex.

**Assessment:** Combines sparsity with competition. May get benefits of both. But still loses magnitude. And introduces non-differentiability at the boundary of the support.

---

## Candidate 7: Temperature-Scaled Softmax

```
a = softmax(z / τ)
```

Temperature τ controls sharpness.

**Forward:** High τ → uniform (weak competition). Low τ → winner-take-all (strong competition).

**Jacobian:** Same structure as softmax, but responsibilities r are computed at temperature τ.

**Gradient flow:** τ modulates competition strength. Learnable τ could adapt during training.

**Competition:** Tunable. τ → 0 gives hard assignment. τ → ∞ gives no competition.

**Expressiveness:** Still constrained to simplex.

**Magnitude:** Still lost.

**Assessment:** Useful knob for controlling competition strength. Doesn't solve magnitude problem. Could be combined with other approaches.

---

## Candidate 8: Softmax Gate on ReLU

```
a = relu(z) ⊙ softmax(z)
```

ReLU for magnitude and sparsity. Softmax for competition.

**Forward:** Active units (z > 0) output their value scaled by responsibility. Inactive units (z ≤ 0) output zero.

**Jacobian:** Complex interaction of ReLU indicator and softmax derivatives.

For zᵢ > 0, zⱼ > 0:
```
∂aᵢ/∂zᵢ = rᵢ + zᵢrᵢ(1 − rᵢ)
∂aᵢ/∂zⱼ = −zᵢrᵢrⱼ
```

For zᵢ ≤ 0: ∂aᵢ/∂zⱼ = 0 for all j.

**Gradient flow:** Inactive units are dead (like ReLU). Active units compete (like softmax-weighted).

**Competition:** Among active units only.

**Expressiveness:** Sparsity from ReLU. Competition among survivors.

**Magnitude:** Preserved for active units, scaled by responsibility.

**Assessment:** Hybrid that may capture benefits of both. Dead units remain dead—no competition can revive them. This could be a feature (sparse) or bug (capacity loss).

---

## Summary Table

| Activation | Competition | Magnitude | Expressiveness | Gradient Complexity |
|------------|-------------|-----------|----------------|---------------------|
| ReLU | None | Partial (≥0) | High (2^K) | Simple |
| Softmax | Strong | Lost | Low (K) | Moderate |
| Log-softmax | Moderate | Relative | Moderate | Simple |
| z ⊙ softmax(z) | Conditional | Scaled | Moderate | Moderate |
| z + softmax(z) | Weak | Full | High | Moderate |
| Grouped softmax | Local | Lost in groups | Tunable | Block diagonal |
| Sparsemax | Strong + sparse | Lost | Moderate | Piecewise |
| Temp softmax | Tunable | Lost | Low | Moderate |
| relu ⊙ softmax | Among active | Scaled | Moderate | Complex |

---

## What's Missing

None of these candidates clearly dominate. The core tension:

- **Competition requires normalization.** Normalization loses magnitude or absolute scale.
- **Magnitude preservation avoids normalization.** Without normalization, no competition.

Possible resolutions:

1. **Accept magnitude loss.** Maybe downstream layers don't need magnitude. Test empirically.

2. **Parallel paths.** Compute competitive and non-competitive activations separately. Concatenate. Let downstream layers use both signals.

3. **Normalization in a subspace.** Compete over direction, preserve magnitude separately. Like: a = ||z|| · softmax(z/||z||). Magnitude in the norm, competition in the direction.

4. **Learned tradeoff.** Let the network learn how much competition to apply. Gating between ReLU and softmax paths.

5. **New activation not yet considered.** The right answer may not be on this list.

---

## Experimental Priority

If testing, order by likely insight:

1. **Softmax** — Maximum competition baseline. Does it specialize? Does it train at all?

2. **Log-softmax** — Moderate competition, some magnitude. Practical middle ground?

3. **z ⊙ softmax(z)** — Novel hybrid. Competition with magnitude. Stability unknown.

4. **Grouped softmax** — Tunable competition. Can we find the right group size?

5. **Others** — As needed based on results.