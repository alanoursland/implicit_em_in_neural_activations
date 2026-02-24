# Paper Outline: Testing Implicit EM Theory in Supervised Networks

## Title candidates

- Volume Control in Supervised Networks: Testing Implicit EM at Intermediate Layers
- The ImplicitEM Layer: Testing Volume Control Theory in Supervised Networks
- (defer final title until results are in)

## Abstract

Paper 1 showed that gradient descent on LSE objectives performs implicit EM. Paper 2 validated this in the unsupervised regime: a model built from the theory's requirements worked, and predicted failure modes appeared when components were removed. The theory also predicts that volume control — the neural analogue of the log-determinant in mixture models — is needed wherever implicit EM operates. In supervised networks, labels provide volume control at the output layer but not at intermediate layers. We test this prediction. We construct an ImplicitEM layer — distance computation, volume control, and probabilistic calibration — and place it as the intermediate layer of a supervised classifier on MNIST. An ablation study removes volume control components and measures intermediate layer health. [Results sentence — deferred until experiments complete.] [Conclusion sentence — deferred.]

## 1. Introduction

### The state of the theory

Paper 1 derived the gradient-responsibility identity. Paper 2 tested the unsupervised regime. Two of three claimed regimes have been addressed (unsupervised and, by Paper 1's analysis, the output layer of supervised networks). The intermediate layers of supervised networks remain untested.

### The gap

The theory predicts that volume control is needed wherever implicit EM operates. Labels provide it at the output. Nothing provides it at intermediate layers. Standard heuristics (BatchNorm, weight decay, dropout) partially compensate but are not derived from the theory. Whether intermediate layers actually need explicit volume control — or whether supervised gradients suffice — is an empirical question.

### This paper

We test the theory's prediction. We derive an ImplicitEM layer from the theory's requirements. We place it in a supervised network. We ablate its components and observe whether the predicted failure modes appear.

## 2. Background

Brief. Readers have Paper 1 and Paper 2 as references.

### 2.1 The gradient-responsibility identity

One paragraph. State ∂L/∂dⱼ = −rⱼ. Reference Paper 1.

### 2.2 Volume control

One paragraph. The log-determinant in GMMs. Its decomposition into variance (anti-collapse) and decorrelation (anti-redundancy). Paper 2's validation: LSE alone collapses, InfoMax prevents it. Reference Paper 2.

### 2.3 Where EM lives

The key theoretical setup. EM exists only where exponentiation + normalization occurs. In standard supervised networks, this is the output layer (cross-entropy + softmax). Internal layers with elementwise activations have diagonal Jacobians — no competition, no EM. The supervised gradient flows back but does not create EM structure at intermediate layers.

### 2.4 The prediction

Labels provide volume control at the output (anti-collapse: every class receives labeled gradient; anti-redundancy: labels break symmetry between classes). Labels do not provide volume control at intermediate layers. The theory predicts the same failure modes from the unsupervised case should appear at intermediate layers when volume control is absent, even with supervised gradients flowing through.

## 3. The ImplicitEM Layer

Derived from theory. Each component traced to a requirement.

### 3.1 Architecture

```
x → Linear → Softplus → [volume control] → NegLogSoftmin → output
```

- Linear: projection onto K learned directions (hypotheses)
- Softplus: logistic kernel, converts projections to non-negative distances
- Volume control: LSE + variance + decorrelation (auxiliary loss on distances)
- NegLogSoftmin: calibration, distance in → distance out, absorbs partition function

### 3.2 NegLogSoftmin

Derivation as type-preserving calibration: distance → probability → log-probability → distance. The result: yⱼ = dⱼ + log Z. Preserves relative distances. Calibrates absolute values so exp(-yⱼ) = rⱼ.

### 3.3 Volume control

Same as Paper 2. LSE loss + variance penalty + decorrelation penalty applied to raw distances d. Neural analogue of the log-determinant.

### 3.4 Complete model

ImplicitEM layer + Linear + LayerNorm + CE. Two loss terms: supervised (CE) and auxiliary (volume control). Total loss = CE + λ · auxiliary.

## 4. Experiment

### 4.1 Setup

MNIST. 64 intermediate components. Six ablation configurations. Three seeds per config. Training protocol. Metrics: dead units, redundancy, responsibility entropy, classification accuracy.

### 4.2 Results

The ablation table. Direct comparison with Paper 2 Table 5.

### 4.3 Feature visualization

Weight images for all six configs. One figure.

## 5. Discussion

### If predictions confirmed

The theory extends to supervised networks. Intermediate layers need volume control independently of supervision. The ablation pattern replicates Paper 2's unsupervised results inside a supervised network. Brief comment on what this implies about standard heuristics (BatchNorm etc.) as implicit volume control — one paragraph, not a full analysis.

### If predictions disconfirmed

The theory's scope is narrower. Supervised gradients provide something the theory doesn't account for. Discuss what that might be and what it means for the framework.

### Limitations

Same as Paper 2: MNIST only, single intermediate layer, no LLM activations, no formal EM convergence proof. Additionally: one value of λ, Softplus only (no kernel comparison).

## 6. Conclusion

The theory predicted specific failure modes at intermediate layers. We tested those predictions. [Result.] [Implication for the theory's scope.]

## Estimated Length

- Introduction: 1.5 pages
- Background: 1.5 pages
- ImplicitEM Layer: 2 pages
- Experiment: 2 pages (table + figure + interpretation)
- Discussion: 1 page
- Conclusion: 0.5 pages
- References: 1 page

Total: ~10 pages. Comparable to Paper 2 (22 pages but with more experiments and extensive discussion). This is leaner because the theory is already established and we test one prediction.