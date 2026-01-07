# Encoder Study: Paper Overview

## The Core Idea

We derive decoder-free sparse autoencoders from first principles using implicit expectation-maximization theory. The derivation is the contribution. The architecture at the end is not new—what's new is understanding *why* it works.

## The Problem

Sparse autoencoders learn interpretable features from neural network activations. The standard architecture is:

```
Encoder: a = ReLU(Wx + b)
Decoder: x̂ = W'a
Loss: ||x - x̂||² + λ||a||₁
```

This works, but requires a decoder. The decoder doubles parameters and compute. More importantly, it raises a question: what is the decoder actually doing? Is it necessary, or is it compensating for something missing in the encoder objective?

## The Insight

Two prior results set up the derivation:

**Result 1 (Oursland 2024):** Linear layers compute distances to prototypes. Each row of W is a prototype direction. The bias determines the prototype location. Neural networks can be interpreted as computing Mahalanobis distances to learned Gaussian components.

**Result 2 (Oursland 2025):** For any objective with log-sum-exp structure L = -log Σⱼ exp(-Eⱼ), the gradient with respect to each energy equals its responsibility:

```
∂L/∂Eⱼ = rⱼ where rⱼ = softmax(-E)
```

This is exact, not approximate. Gradient descent on LSE objectives performs EM implicitly. The E-step (computing responsibilities) happens in the backward pass. The M-step (updating parameters) happens in the optimizer step.

**The Gap:** LSE alone collapses. One component claims all inputs, others die. This is a known problem in mixture models. Gaussian mixtures prevent collapse via the log-determinant (volume penalty). Neural networks have no equivalent term.

**Our Contribution:** We identify InfoMax regularization as the missing volume control. The variance term prevents dead components. The decorrelation term prevents redundant components. Together, they serve the role of the log-determinant in mixture models.

## The Derivation

Combining LSE (for EM structure) with InfoMax (for volume control) yields:

```
L = -log Σⱼ exp(-aⱼ) + λ_var(-Σⱼ log Var(aⱼ)) + λ_tc||Corr(A) - I||²
```

This is a complete unsupervised objective for learning sparse features. No decoder. No reconstruction. The encoder learns to:

1. Cover the data (LSE term: at least one component should explain each input)
2. Use all components (variance term: every component must be active)
3. Learn distinct components (correlation term: components must differ)

The result is a decoder-free sparse autoencoder derived from first principles.

## Why It Works (Intuition)

The LSE and InfoMax terms create opposing forces:

**LSE is repulsive.** Minimizing -log Σ exp(-a) penalizes components that are too close to inputs relative to others. It distributes responsibility.

**InfoMax is attractive.** Maximizing variance requires components to respond strongly to some inputs. It pulls components toward the data.

The equilibrium: components spread out to cover the data, each specializing in a different region. This is competitive learning / vector quantization, but differentiable and probabilistic.

## What We Claim

1. **The derivation is valid.** LSE + InfoMax produces a principled encoder objective.
2. **It prevents collapse.** Without InfoMax, pure LSE degenerates.
3. **It works.** The resulting model matches standard SAEs on benchmarks.

## What We Don't Claim

- That this is better than existing SAEs
- That we achieve state-of-the-art on any metric
- That the "Component Field" interpretation is rigorously formalized
- That we've solved interpretability

The contribution is theoretical: we explain where decoder-free SAEs come from. Empirical validation confirms the theory works. We don't need to prove superiority.

## The Experiments

Three experiments validate three claims:

**Experiment 1: Verify the Theorem**

Record ∂L_LSE/∂aⱼ and rⱼ during one backward pass. Plot them against each other. Should be identity line. This proves the implicit EM identity is implemented correctly.

**Experiment 2: Ablation**

Train four configurations:
- LSE only → expect collapse
- LSE + variance → expect no dead units, but redundancy
- LSE + variance + correlation → expect stable diverse features
- InfoMax only (no LSE) → expect different behavior (no EM structure)

This proves each term serves its claimed purpose.

**Experiment 3: Benchmark**

Compare to standard SAE on MNIST or LLM activations. Measure reconstruction (using W^T as implicit decoder), sparsity, and parameters. Should be competitive with half the parameters.

This proves the model is practical.

## The Narrative Arc

1. **Setup:** Sparse autoencoders are useful but require decoders.

2. **Question:** Can we derive an encoder-only objective from first principles?

3. **Background:** LSE objectives perform implicit EM. Linear layers compute distances to prototypes.

4. **Problem:** LSE alone collapses. Mixture models prevent this with volume penalties. Neural networks lack equivalent.

5. **Solution:** InfoMax regularization provides volume control. Variance prevents death. Decorrelation prevents redundancy.

6. **Result:** LSE + InfoMax = decoder-free sparse autoencoder.

7. **Validation:** Three experiments confirm the theory.

8. **Conclusion:** The decoder was never necessary. It was compensating for missing volume control in the encoder objective. With proper regularization, the encoder alone suffices.

## Connections to Prior Work

**Energy-Based Models (LeCun):** Our LSE term is the negative log marginal likelihood under a mixture of energy-based components. We add InfoMax as regularization on the energy landscape.

**Sparse Coding (Olshausen & Field):** Traditional sparse coding uses reconstruction + L1. We use LSE + InfoMax. No decoder, no explicit sparsity penalty. Sparsity emerges from competition.

**InfoMax/ICA (Bell & Sejnowski):** We borrow the operational definition (maximize variance, minimize correlation) but don't claim to compute mutual information exactly. InfoMax is a tool, not a theorem.

**Competitive Learning (Kohonen):** The dynamics resemble self-organizing maps. LSE repels, InfoMax attracts, equilibrium is coverage. The difference: we're fully differentiable with soft responsibilities.

## The Key Equations

**Implicit EM Identity:**
```
∂/∂Eⱼ[-log Σₖ exp(-Eₖ)] = softmax(-E)ⱼ = rⱼ
```

**Complete Objective:**
```
L = -log Σⱼ exp(-aⱼ) - λ_var Σⱼ log Var(aⱼ) + λ_tc ||Corr(A) - I||²
```

**Architecture:**
```
z = Wx + b
a = relu(z)
L = LSE(a) + InfoMax(a)
```

## Success Criteria

The paper succeeds if:

1. A reader understands why decoder-free SAEs work (the derivation)
2. The experiments confirm the theoretical predictions
3. The connection to implicit EM is clear and believable

The paper fails if:

1. It claims novelty for the architecture (which exists)
2. It oversells empirical results (we need competitiveness, not superiority)
3. It overcomplicates the story (the derivation should feel inevitable)

## One-Sentence Summary

Decoder-free sparse autoencoders arise naturally from implicit EM when log-sum-exp objectives are combined with InfoMax regularization, which serves as volume control analogous to the log-determinant in Gaussian mixture models.