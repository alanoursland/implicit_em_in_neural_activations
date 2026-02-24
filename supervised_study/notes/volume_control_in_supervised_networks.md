# Volume Control in Supervised Networks

## The Central Argument

Labels provide volume control at the output layer. Nothing provides it at intermediate layers. This asymmetry is the theoretical foundation of the paper.

## What Volume Control Is

In Gaussian mixture models, the log-likelihood of a data point under component k includes:

```
log P(x | k) ∝ -½(x - μₖ)ᵀΣₖ⁻¹(x - μₖ) - ½ log det(Σₖ)
```

The first term is the Mahalanobis distance — it rewards placing the component close to data. The second term is the log-determinant — it penalizes shrinking the covariance. Without it, a component can collapse to a point and achieve unbounded likelihood.

The log-determinant serves two functions:

**Anti-collapse (diagonal).** Each component must maintain non-zero variance. As any eigenvalue of Σₖ approaches zero, the log-determinant diverges to -∞. A component cannot claim a zero-volume region.

**Anti-redundancy (off-diagonal).** Correlated components reduce the determinant. If two components become identical, the joint covariance matrix becomes singular and the determinant vanishes. Components must be distinct.

Volume control = anti-collapse + anti-redundancy. Both are needed for a healthy mixture.

## Volume Control at the Output Layer

Consider a K-class classification network trained with cross-entropy. The output layer has K logits, one per class. Softmax normalizes them to responsibilities. The label clamps the correct class to responsibility 1.

### Labels prevent collapse.

Every class has labeled examples. For each example with label y, the correct class receives gradient:

```
∂L/∂zᵧ = rᵧ - 1
```

As long as the model isn't perfect (rᵧ < 1), the correct class receives gradient. It cannot die. Even a class with only a few examples receives gradient on every one of those examples. The labels guarantee that every class participates in learning.

This is anti-collapse. The labeled data distribution ensures every component receives gradient signal proportional to its frequency in the dataset.

### Labels prevent redundancy.

If two classes had identical weight vectors, they would produce identical logits for all inputs. Softmax would assign them equal responsibility. But labeled examples for class A would push class A's weights in one direction while labeled examples for class B would push class B's weights in a different direction (toward B's data). The asymmetry of the labels forces separation.

More precisely: the gradient for incorrect classes is rⱼ (pushed away in proportion to responsibility). If classes A and B are identical, they split responsibility equally. But class A gets pulled toward A-labeled data and pushed away from B-labeled data, while class B gets the opposite. The label asymmetry breaks the symmetry of identical weights.

This is anti-redundancy. Labels provide distinct targets that force components apart.

### Labels are sufficient at the output.

For a well-posed classification problem (distinct classes, reasonable label distribution), the output layer does not need explicit volume control. No variance penalty. No decorrelation penalty. No log-determinant. The labels do the work.

This is why standard cross-entropy classification works without any InfoMax-style regularization at the output. The supervision is the volume control.

## Volume Control at Intermediate Layers

Now consider an intermediate layer with K hidden units. No labels. No direct supervision. Only gradients flowing back from the output.

### The supervised gradient does not prevent collapse.

The gradient to intermediate unit j depends on how unit j affects the final loss. If unit j contributes nothing — its activation is constant, or its weight to the output is zero — its gradient is zero. It receives no learning signal. It dies.

But unlike the output layer, no label says "unit j must be active for this input." The supervised gradient is indifferent to which intermediate units are alive, as long as the output is correct. If 10 of 64 intermediate units can produce a good classification, the other 54 can die without consequence to the loss.

This is the collapse problem. The supervised gradient provides no floor on intermediate unit activity. Units can and do die.

### The supervised gradient does not prevent redundancy.

If units 3 and 7 learn identical features, both contribute equally to the output. The gradient to both is identical. Nothing pushes them apart. They remain redundant.

The output layer's labels break symmetry between classes because different labels pull different classes in different directions. But intermediate units have no labels. If two units are identically useful to the output, they receive identical gradients. Symmetry is preserved. Redundancy persists.

This is the redundancy problem. The supervised gradient provides no pressure for intermediate units to differentiate.

### The supervised gradient does not care about geometry.

The supervised gradient wants "produce features that help classify." It does not want "produce features that form a well-conditioned mixture model." If a degenerate representation — collapsed dimensions, correlated features, dead units — happens to support good classification, the supervised gradient is satisfied.

In practice, degenerate intermediate representations are common:

- Many units learn overlapping features (redundancy)
- Some units have near-zero variance (partial collapse)
- The effective dimensionality is much lower than the nominal width
- Networks can be pruned to 10% of parameters with minimal accuracy loss

These are symptoms of missing volume control at intermediate layers.

## What Provides Volume Control in Practice

Standard deep learning uses heuristics that happen to provide partial volume control:

### Weight decay

Penalizes ||W||². Prevents unbounded weight growth. This limits the scale of projections, preventing one unit from dominating by having larger weights. But it penalizes all weights equally — it does not specifically target collapse or redundancy.

**As volume control:** Weak. Addresses scale but not structure. Two identical units with small weights are still redundant. A unit with small but nonzero weights can still be effectively dead if it falls in a flat region.

### BatchNorm

Normalizes activations per feature across the batch to zero mean and unit variance. Forces each unit to have non-trivial activation statistics.

**As volume control:** Partial. The variance normalization directly prevents collapse — a unit with zero variance would produce division by zero. But BatchNorm does not prevent redundancy. Two units with identical activations (perfect correlation) both pass through BatchNorm unchanged.

BatchNorm addresses the diagonal of the log-determinant (variance) but not the off-diagonal (correlation).

### LayerNorm

Normalizes activations per sample across features. Standardizes the representation scale for each input.

**As volume control:** Weak for intermediate layers. LayerNorm normalizes the total activation magnitude but does not constrain individual units. A layer where one unit dominates and all others are near zero would be normalized to have unit variance, but the dominance structure is preserved.

### Dropout

Randomly zeros units during training. Forces the network to distribute information across units — no single unit can be essential.

**As volume control:** Indirect anti-collapse. A unit that is frequently dropped cannot be the sole carrier of information. Other units must compensate. This distributes learning signal. But dropout is stochastic and does not directly target redundancy or geometric health.

### Residual connections

Skip connections add the input to the output: y = x + f(x). The layer only needs to learn a residual. The identity path preserves signal even if f(x) is degenerate.

**As volume control:** Prevents complete signal loss (a form of anti-collapse for the representation as a whole). But does not constrain the structure of f(x) itself. The residual pathway can be redundant or collapsed without affecting the skip connection.

### Summary of heuristics

| Mechanism | Anti-collapse | Anti-redundancy | Principled |
|-----------|:---:|:---:|:---:|
| Weight decay | Weak | ✗ | ✗ |
| BatchNorm | ✓ (variance) | ✗ | ✗ |
| LayerNorm | Weak | ✗ | ✗ |
| Dropout | Indirect | Indirect | ✗ |
| Residual connections | Signal preservation | ✗ | ✗ |
| **Labels (output only)** | **✓** | **✓** | **✓** |
| **InfoMax (explicit)** | **✓ (variance)** | **✓ (decorrelation)** | **✓** |

None of the standard heuristics provides complete volume control at intermediate layers. Each addresses part of the problem. Labels provide complete volume control but only at the output. InfoMax (variance + decorrelation) provides complete, principled volume control wherever it is applied.

## The ImplicitEM Layer Provides What Is Missing

The ImplicitEM layer applies InfoMax regularization to intermediate distances:

```
L_var = -Σ log Var(dⱼ)        → anti-collapse (diagonal of log-determinant)
L_tc = ||Corr(d) - I||²_F     → anti-redundancy (off-diagonal of log-determinant)
```

These are the neural equivalents of the log-determinant, as established in Paper 2. They provide the same guarantees:

- Every component must maintain non-zero variance across the batch → no dead units
- Components must respond differently across inputs → no redundant units

Combined with the LSE loss (which provides EM dynamics — attraction toward data, responsibility-weighted competition), the result is a well-formed mixture at the intermediate layer.

## The Asymmetry Restated

| Layer | EM dynamics | Volume control source | Complete? |
|-------|:-----------:|:--------------------:|:---------:|
| Output (CE) | From cross-entropy (label-clamped) | From labels | ✓ |
| Intermediate (standard) | None (diagonal Jacobian) | From heuristics (weight decay, BN, etc.) | ✗ |
| Intermediate (ImplicitEM) | From auxiliary LSE loss | From InfoMax (variance + decorrelation) | ✓ |

The output layer is fully specified by the standard supervised setup. It has EM from cross-entropy and volume control from labels. Nothing needs to be added.

The intermediate layer, in a standard network, has neither EM nor volume control. Heuristics partially compensate. The ImplicitEM layer provides both explicitly.

## What This Predicts

### Without volume control, intermediate layers degenerate.

Even with supervised gradients flowing through, intermediate units should show:
- Dead units (some components have near-zero variance)
- Redundancy (high correlation between components)
- Unstructured weights (no interpretable prototype character)

This is the baseline MLP in our ablation (Config 1).

### With volume control, intermediate layers form mixtures.

The ImplicitEM layer should produce:
- Zero dead units (variance penalty prevents collapse)
- Low redundancy (decorrelation penalty prevents correlation)
- Interpretable weights (mixture components, digit prototypes)

This is the full ImplicitEM in our ablation (Config 5).

### The ablation should replicate Paper 2.

The intermediate layer is unsupervised from the volume control perspective. The same failure modes that appeared in the unsupervised model should appear here:
- LSE alone collapses (no volume control)
- Variance prevents death but not redundancy
- Decorrelation prevents redundancy
- Full InfoMax produces healthy mixture

The supervised gradient flowing through is an additional signal, but it does not substitute for volume control. It may modulate the results (e.g., classification-relevant components may be favored) but it should not eliminate the need for explicit anti-collapse and anti-redundancy measures.

## The Broader Implication

Every intermediate layer in every deep network faces the volume control problem. The standard solution is a stack of heuristics: BatchNorm for scale, weight decay for magnitude, dropout for distribution, residual connections for signal preservation. These work in practice but are not derived from any unified theory.

The implicit EM framework provides that theory. Volume control is the neural analogue of the log-determinant. It is needed wherever components compete — at the output (provided by labels) and at every intermediate layer (not provided by anything in standard architectures).

The ImplicitEM layer is a proof of concept: principled volume control at an intermediate layer, derived from mixture model theory, applied inside a supervised network. If it works as predicted, the implication extends beyond this specific architecture: the heuristics that stabilize deep learning are approximations to volume control, and the theory tells us what they're approximating.