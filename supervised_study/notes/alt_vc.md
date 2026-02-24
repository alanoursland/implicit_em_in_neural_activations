# Alternative Volume Control: Standard Techniques Reinterpreted

## Purpose

These are notes, not paper content. The paper will contain one paragraph in the discussion observing that standard heuristics partially substitute for volume control. This document is the thinking behind that paragraph.

## The Framework

Volume control has two functions:

- **Anti-collapse (diagonal):** Every component must maintain non-zero variance. No dead units.
- **Anti-redundancy (off-diagonal):** Components must respond differently. No duplicates.

In GMMs, the log-determinant provides both. In the ImplicitEM layer, variance + decorrelation provide both. The question: do standard deep learning techniques provide either?

## Batch Normalization

### What it does

Per-feature normalization across the batch:

```
â_j = (a_j - μ_j) / (σ_j + ε)
ŷ_j = γ_j · â_j + β_j
```

Each feature is centered to zero mean and scaled to unit variance, then rescaled by learned parameters γ, β.

### As volume control

**Anti-collapse: Yes, directly.** If a feature has zero variance (σ_j = 0), the normalization divides by ε, producing large values. The gradient through this pathological normalization is enormous. In practice, this means a feature cannot maintain zero variance — the gradient forces it to activate. BatchNorm makes dead units mathematically unstable.

This is the diagonal of the log-determinant. BatchNorm enforces non-zero variance per feature.

**Anti-redundancy: No.** If features 3 and 7 are perfectly correlated (identical up to scale and shift), BatchNorm normalizes each independently. After normalization, they are still perfectly correlated — just both centered and scaled. The off-diagonal structure is untouched.

Two identical features pass through BatchNorm and remain identical.

### Summary

BatchNorm provides the diagonal but not the off-diagonal. Half of volume control.

## Layer Normalization

### What it does

Per-sample normalization across features:

```
â_i = (a_i - μ) / (σ + ε)     where μ, σ are computed across features for this sample
```

### As volume control

**Anti-collapse: Weak.** LayerNorm normalizes the total activation vector, not individual features. If one feature dominates (large magnitude) while others are near zero, LayerNorm scales the whole vector. The dominant feature gets smaller, the zero features get... still zero, just relative to a different mean. A dead feature contributes nothing to the per-sample statistics and is not rescued.

**Anti-redundancy: No.** Same argument as BatchNorm. Per-sample normalization does not see correlations between features.

### Summary

LayerNorm prevents total signal collapse (the representation as a whole maintains unit scale) but does not address individual feature health. Weaker than BatchNorm for volume control purposes.

## Weight Decay

### What it does

L2 penalty on parameters: L_wd = λ||W||²

### As volume control

**Anti-collapse: Indirect, and in the wrong direction.** Weight decay pushes weights toward zero. A feature with small weights produces small activations with small variance. Weight decay makes collapse *more* likely, not less. It prevents the opposite problem — unbounded weight growth — which is related to winner-take-all collapse in competitive settings (one component grows its weights to dominate softmax). But for non-competitive layers (ReLU), weight decay doesn't prevent dead units.

**Anti-redundancy: No.** Two identical weight vectors have the same L2 norm as two distinct vectors of the same magnitude. Weight decay penalizes magnitude, not structure. Redundant features are not penalized.

### Summary

Weight decay addresses scale but not structure. It prevents one specific collapse mode (unbounded growth) while potentially encouraging another (shrinkage to zero). Not volume control in the mixture model sense.

## Dropout

### What it does

Randomly zeros features during training with probability p. Remaining features scaled by 1/(1-p).

### As volume control

**Anti-collapse: Indirect.** If a feature is dropped, the network must function without it. Other features must carry the information. This distributes information across features — no single feature is essential. A feature that was going to die might receive gradient when a competing feature is dropped.

But this is stochastic and indirect. Dropout doesn't guarantee that dead features recover. It reduces the probability of extreme concentration but doesn't enforce a floor on variance.

**Anti-redundancy: Indirect.** If features 3 and 7 are identical, dropping one has no effect (the other compensates). So redundancy is tolerated, not penalized. In fact, dropout may *encourage* redundancy — redundant features provide natural robustness to dropout, since losing one copy doesn't lose information.

### Summary

Dropout distributes information but doesn't enforce diversity. Its effect on volume control is indirect and stochastic. It's a regularizer that happens to oppose some collapse modes, not a principled volume mechanism.

## Residual Connections

### What it does

y = x + f(x). The layer output includes a skip connection from the input.

### As volume control

**Anti-collapse: For the representation, not the layer.** If f(x) collapses completely (all zeros), y = x still preserves the input signal. The *representation* doesn't collapse. But the *layer* f(x) has collapsed — it contributes nothing. Residual connections prevent the downstream effect of collapse without preventing collapse itself.

This is an important distinction. The representation is healthy because the skip connection carries it through. The layer's capacity is wasted. Residual connections mask the collapse problem rather than solving it.

**Anti-redundancy: No.** If multiple units in f(x) are redundant, the skip connection doesn't change that. It just adds the redundant output to x.

### Summary

Residual connections are collapse-tolerant, not collapse-preventing. They ensure the network still functions when layers degenerate, but they don't prevent degeneration. A network with residual connections can have deeply degenerate intermediate layers and still perform well — because the skip connections route around the damage.

This is arguably why residual connections are so effective: they make the network robust to the very volume control failures that pervade intermediate layers.

## Spectral Normalization

### What it does

Constrains the spectral norm (largest singular value) of weight matrices: W ← W / σ_max(W).

### As volume control

**Anti-collapse: Partial.** Bounding the largest singular value prevents one direction from dominating. But it doesn't enforce a floor on the *smallest* singular value. The weight matrix can still be low-rank — many directions can have near-zero singular values. Some components can effectively die.

**Anti-redundancy: No.** Spectral normalization constrains the maximum but not the distribution of singular values. Two identical rows in W have the same spectral norm as two orthogonal rows.

### Summary

Addresses the upper bound on scale but not the lower bound. Prevents explosion but not collapse.

## Orthogonal Initialization

### What it does

Initializes weight matrices as orthogonal: W₀ᵀW₀ = I.

### As volume control

**Anti-collapse: At initialization only.** Components start diverse and well-separated. But training immediately begins moving them. Nothing maintains orthogonality during training. Components can converge and collapse.

**Anti-redundancy: At initialization only.** Same issue. Orthogonal start, but no ongoing pressure to stay orthogonal.

### Summary

Provides volume control at t=0. No volume control at t>0. A one-time intervention, not a sustained mechanism.

## The Pattern

| Technique | Anti-collapse | Anti-redundancy | Sustained | Principled |
|-----------|:---:|:---:|:---:|:---:|
| BatchNorm | ✓ | ✗ | ✓ | ✗ |
| LayerNorm | Weak | ✗ | ✓ | ✗ |
| Weight decay | ✗ (wrong direction) | ✗ | ✓ | ✗ |
| Dropout | Indirect | ✗ | ✓ | ✗ |
| Residual connections | Masks, doesn't prevent | ✗ | ✓ | ✗ |
| Spectral norm | Partial (upper bound) | ✗ | ✓ | ✗ |
| Orthogonal init | ✓ (at t=0) | ✓ (at t=0) | ✗ | ✗ |
| **Labels (output only)** | **✓** | **✓** | **✓** | **✓** |
| **InfoMax** | **✓** | **✓** | **✓** | **✓** |

No standard technique provides anti-redundancy during training. BatchNorm provides anti-collapse. Everything else is partial, indirect, or one-time.

Only labels (at the output) and InfoMax (wherever applied) provide both functions, sustained, from a principled source.

## The Observation for the Paper

One paragraph in the discussion, something like:

"Standard deep learning techniques partially substitute for volume control. BatchNorm enforces non-zero variance per feature (anti-collapse) but does not address correlation between features (anti-redundancy). Weight decay, dropout, and residual connections each address aspects of representation health but none provides complete volume control. Orthogonal initialization provides both functions at the start of training but not during it. These heuristics are effective in practice but are not derived from any unified theory of what representations need. The implicit EM framework identifies the log-determinant — decomposed into variance and decorrelation — as the principled requirement. The success or failure of our ablation tests whether this principled requirement holds at intermediate layers."

That's the paragraph. This document is the analysis behind it.

## A Thought About BatchNorm

BatchNorm is the closest standard technique to principled volume control. It directly enforces non-zero variance per feature. If you added a correlation penalty to BatchNorm, you'd have something very close to InfoMax.

Barlow Twins and VICReg discovered this independently — variance + covariance regularization for self-supervised learning. Paper 2 noted the convergence. BatchNorm was already halfway there. The field has been circling the log-determinant without naming it.

## A Thought About Residual Connections

The reframing of residual connections as collapse-tolerant (rather than collapse-preventing) is subtle but important. Residual networks don't solve the volume control problem. They make the network robust to it. The intermediate layers can degenerate without catastrophic consequences because the skip connections carry the signal.

This explains a puzzling observation: residual networks have highly redundant intermediate representations (many near-identical features) yet perform well. The redundancy is a volume control failure that the skip connections route around. The capacity is wasted but the output is fine.

If volume control were applied at intermediate layers, residual networks might not need as much depth — each layer would contribute meaningfully rather than being partially bypassed. Speculative, but consistent with the theory.