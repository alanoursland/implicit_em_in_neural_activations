# InfoMax vs. Pretraining Prior Art

## The History

Greedy layer-wise pretraining was essential to deep learning's revival. Before 2006, deep networks were notoriously difficult to train. Hinton et al. showed that pretraining each layer as a Restricted Boltzmann Machine, then fine-tuning end-to-end, made deep learning work.

Then it stopped mattering. By 2012, end-to-end training with ReLU, dropout, and better initialization worked fine. Pretraining became unnecessary. The field moved on.

Now pretraining is back, at massive scale. GPT, BERT, CLIP—pretrain on huge data, fine-tune on tasks. The idea that general features transfer is validated.

Where does InfoMax + implicit EM fit in this history?

## RBM Pretraining (Hinton 2006)

**Architecture:** Restricted Boltzmann Machine per layer. Bipartite undirected graphical model.

**Objective:** Log-likelihood of data. Learn P(x).

**Training:** Contrastive divergence. Approximate MCMC. Expensive.

**What it learns:** Features that enable reconstruction. Generative model of each layer's input.

**Why it helped:** Initialized weights in good basins. Prevented vanishing gradients in sigmoid networks. Each layer started with meaningful features.

**Why it died:** ReLU solved gradient flow. BatchNorm stabilized training. Better initialization (Xavier, He) worked. Brute force compute succeeded. The careful pretraining became unnecessary overhead.

## Autoencoders (Vincent et al. 2008)

**Architecture:** Encoder-decoder per layer. Deterministic.

**Objective:** Reconstruction loss. ||x - decode(encode(x))||².

**Training:** Standard backprop. Cheaper than RBMs.

**What it learns:** Features that preserve information. Compression.

**Variants:**
- Denoising autoencoders: reconstruct clean from corrupted
- Sparse autoencoders: encourage sparse codes
- Variational autoencoders: learn latent distribution

**Limitation:** Reconstruction doesn't prevent redundancy. Multiple features can encode the same information. Bottleneck forces compression but not independence.

## Contrastive Learning (2018-2020)

**Architecture:** Encoder + projection head. Siamese or momentum networks.

**Objective:** Attract augmentations of same image, repel different images.

**Training:** Large batches, hard negatives, careful augmentation.

**What it learns:** Features invariant to augmentation, distinctive across instances.

**Examples:** SimCLR, MoCo, BYOL, SwAV.

**Limitation:** Requires augmentation strategy. What augmentations to use? Domain-specific. Also doesn't explicitly prevent redundancy—features can be correlated if they all help the contrastive task.

## Modern Foundation Models (2018-present)

**Architecture:** Transformers. Massive scale.

**Objective:** Next token prediction (GPT), masked prediction (BERT), contrastive (CLIP).

**Training:** Enormous data, enormous compute.

**What it learns:** Features useful for predicting structure in data.

**Why it works:** Scale. Diverse data. The objective is simple but training is massive.

**Limitation:** Redundancy is rampant. Models can be pruned 50-90%. Features are entangled. Interpretability is poor. We succeed by brute force, not efficiency.

## InfoMax + Implicit EM (This Work)

**Architecture:** Linear layer + softmax (or competitive activation).

**Objective:** Maximize H(zⱼ) per output, minimize TC(z) across outputs.

**Training:** Standard backprop with batch statistics. Cheap.

**What it learns:** Independent, informative features. Each output captures unique information about input.

**Mechanism:** Softmax provides responsibility-weighted gradients (implicit EM). InfoMax provides the objective.

## The Comparison

| Approach | Objective | What It Prevents | Computational Cost |
|----------|-----------|------------------|-------------------|
| RBM | Density estimation | Poor initialization | High (sampling) |
| Autoencoder | Reconstruction | Information loss | Low |
| Contrastive | Instance discrimination | Augmentation-variance collapse | Medium (large batches) |
| Foundation models | Prediction | Nothing specific | Extreme |
| InfoMax + EM | Independence | Redundancy, collapse | Low |

## What Each Objective Actually Optimizes

**RBM (density estimation):**
"Find features such that you can model P(x)."

But modeling P(x) doesn't require non-redundant features. A mixture of Gaussians with overlapping components still models the density.

**Autoencoder (reconstruction):**
"Find features such that you can reconstruct x."

Reconstruction allows redundancy. If feature 1 and feature 2 both encode the same information, reconstruction works. One is wasted, but the objective doesn't care.

**Contrastive (discrimination):**
"Find features such that same-instance augmentations are close, different instances are far."

Redundancy doesn't hurt discrimination. If 10 features all encode "is this a dog," the contrastive objective is satisfied. Wasteful but valid.

**InfoMax (independence):**
"Find features such that each one tells you something new."

Redundancy directly hurts the objective. Correlated features increase TC. Penalized. Dead features decrease marginal entropy. Penalized.

InfoMax is the only objective that explicitly targets non-redundancy.

## Why RBM Pretraining Died

The standard story: better tools made it unnecessary.

A deeper story: the objective was wrong.

Density estimation doesn't give you good features. It gives you features sufficient for modeling P(x). These overlap with good features but aren't identical.

End-to-end training on classification also gives you features sufficient for the task. These also overlap with good features.

Both approaches get you into the right neighborhood. Neither optimizes for what actually makes features good.

What makes features good:
- Each captures a distinct factor of variation
- Together they span the relevant structure
- No redundancy, no gaps

This is independence. InfoMax optimizes for it directly.

## Why End-to-End Training Works (Mostly)

End-to-end training finds *some* good features. The task requires certain distinctions. Features that make those distinctions are learned.

But:
- Features not needed for the task aren't learned (gaps)
- Features can be redundant if redundancy doesn't hurt the task (waste)
- Features are task-specific, not universal

The result: models work but are inefficient. Massive redundancy. Poor transfer. Need to retrain for each task.

We compensate with scale. More parameters to search. More data to constrain. It works, but wastefully.

## The InfoMax Hypothesis

Good features are independent factors of variation.

If you find them, they're useful for everything. Classification, generation, segmentation, transfer—all become easy. The features *are* understanding.

End-to-end training finds them partially, redundantly, task-specifically.

InfoMax finds them directly.

**Predictions:**

1. InfoMax-pretrained models can't be pruned. Each feature carries unique information. No redundancy to remove.

2. InfoMax features transfer better. They're universal, not task-specific.

3. Smaller InfoMax models match larger conventional models. Same information, fewer parameters.

4. InfoMax features are interpretable. Each corresponds to an independent factor. You can name them.

## What InfoMax Gets from EM

InfoMax is the objective. EM is the mechanism.

Why does the mechanism matter?

**Without EM (just InfoMax on raw activations):**
- Features compete for information
- But no structure to the competition
- Gradient descent finds *a* solution
- May be unstable, initialization-dependent

**With EM (softmax responsibilities):**
- Features compete via soft assignment
- Each input is probabilistically allocated
- Responsibility-weighted updates
- Stable, structured learning dynamics

EM provides the inductive bias. Features aren't just uncorrelated—they're prototypes that own regions of input space. This is a stronger structure than just "be independent."

## What Remains from Prior Art

**From RBMs:** The idea that local unsupervised objectives can initialize good features.

**From autoencoders:** The simplicity. No sampling, just backprop.

**From contrastive learning:** Batch-based objectives can work. Coupling samples is okay.

**From foundation models:** Pretraining transfers. Universal features exist.

**New from InfoMax + EM:** Explicit optimization for non-redundancy. Principled volume control. Theoretical grounding in information theory and probabilistic inference.

## Open Questions

**Does InfoMax scale?**
RBMs didn't scale. Autoencoders did but weren't enough. Contrastive learning scales. Foundation model objectives scale massively. Where does InfoMax land?

**Is independence the right target?**
Maybe some redundancy is good (robustness). Maybe some correlation captures real structure. InfoMax assumes independence is optimal. Is it?

**Does EM structure matter at scale?**
With enough parameters and data, does the learning mechanism matter? Or does everything converge to similar features eventually?

**How does InfoMax interact with depth?**
Layer 1 produces independent features of pixels. What does layer 2 produce? Independent features of independent features? Does this compose sensibly?

These questions need experiments.