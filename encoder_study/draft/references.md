# Related Work and Theoretical Context

This document consolidates and organizes the full set of applicable citations underlying the proposed decoder-free, implicit-EM, InfoMax-regularized framework. It unifies the original seed list with additional references identified in *infomax_as_em_objective.md* and *infomax_vs_pretraining_prior_art.md*, removes redundancy, and presents a coherent narrative suitable for a **Related Work**, **Theory**, or **Discussion** section of an arXiv paper.

The emphasis throughout is on *why* each citation is necessary, not merely that it is thematically related.

---

## 1. Core Theory: First Principles

### Implicit EM and Distance-Based Learning

**Oursland, A. (2025). *Gradient Descent as Implicit EM in Distance-Based Neural Models*. arXiv:2512.24780.**

This paper provides the central theoretical foundation. It establishes an exact algebraic identity showing that gradient descent on log-sum-exp (LSE) objectives implicitly performs Expectation–Maximization: gradients correspond to soft responsibilities (E-step), while parameter updates correspond to prototype updates (M-step). The present work builds directly on this result, supplying the missing objective-level constraints—via InfoMax regularization—that make implicit EM viable for unsupervised representation learning without collapse.

---

**Oursland, A. (2024). *Interpreting Neural Networks through Mahalanobis Distance*. arXiv:2410.19352.**

This work provides the geometric interpretation that linear layers compute Mahalanobis distances to learned prototypes. This reframing motivates treating encoder activations as *energies* rather than similarities, justifying the use of LSE as a marginal likelihood. It also predicts the necessity of orthogonality or redundancy control on weights, which reappears in the present work as part of the full volume-control regularization.

---

### Energy-Based Modeling

**LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M., & Huang, F. J. (2006). *A tutorial on energy-based learning*. In *Predicting Structured Data*.**

This tutorial formalizes learning via energy functions rather than explicit normalized probability models. The LSE term used here is precisely the negative log marginal likelihood of a mixture of energy-based components. The InfoMax regularization can be interpreted as shaping the energy landscape to prevent degenerate minima, situating this work squarely within the energy-based modeling tradition.

---

## 2. The Objective: InfoMax and Volume Control

### Information Maximization

**Bell, A. J., & Sejnowski, T. J. (1995). *An information-maximization approach to blind separation and blind deconvolution*. Neural Computation.**

This work introduced InfoMax as a principle for learning independent components by maximizing output entropy and minimizing redundancy. The present framework borrows the *operational* insight rather than the exact mutual-information formulation: variance maximization and decorrelation are used as practical proxies for independence. InfoMax here functions as volume control for implicit EM, rather than as a complete statistical objective.

---

**Linsker, R. (1988). *Self-organization in a perceptual network*. Computer.**

Linsker demonstrated that maximizing information transmission alone can lead to the spontaneous emergence of structured, meaningful representations. The present work follows this philosophy: useful features arise from information-theoretic constraints rather than task supervision. This work extends the idea by embedding InfoMax inside an implicit EM mechanism, giving structure to the learning dynamics rather than relying on unconstrained gradient ascent.

---

### Redundancy Reduction and Collapse Prevention

**Zbontar, J., Jing, L., Misra, I., LeCun, Y., & Deny, S. (2021). *Barlow Twins: Self-Supervised Learning via Redundancy Reduction*. ICML.**

This work introduces a tractable decorrelation objective based on cross-correlation matrices. The decorrelation proxy used in the present framework is mathematically equivalent. Although developed for contrastive-free self-supervised learning, the same insight applies: redundancy reduction is essential to prevent representational collapse.

---

**Bardes, A., Ponce, J., & LeCun, Y. (2022). *VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning*. ICLR.**

VICReg explicitly decomposes self-supervised objectives into variance, invariance, and covariance terms. The variance and decorrelation terms used here correspond directly to VICReg’s variance and covariance penalties, providing strong independent evidence that these regularizers are broadly effective as collapse-prevention mechanisms.

---

## 3. Architecture and Learning Dynamics

### Sparse Coding and Its Simplification

**Olshausen, B. A., & Field, D. J. (1996). *Emergence of simple-cell receptive field properties by learning a sparse code for natural images*. Nature.**

Sparse coding demonstrated that interpretable features can emerge from unsupervised learning when sparsity is enforced via reconstruction and an L1 penalty. The present work achieves similar outcomes—sparse, interpretable features—but removes the decoder entirely. Sparsity emerges from competitive responsibility allocation and ReLU nonlinearities, reframing sparse coding as an implicit EM problem with volume control rather than explicit reconstruction.

---

### Competitive Learning

**Kohonen, T. (1982). *Self-organized formation of topologically correct feature maps*. Biological Cybernetics.**

Kohonen’s self-organizing maps demonstrate how prototypes distribute themselves to cover input space through attraction to data and mutual competition. The dynamics recovered here are closely related but arise in a fully differentiable, probabilistic framework. Soft responsibilities replace hard winner-take-all updates, and InfoMax regularization replaces heuristic repulsion terms.

---

## 4. EM, Mixture Models, and Collapse

**Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). *Maximum likelihood from incomplete data via the EM algorithm*. Journal of the Royal Statistical Society: Series B.**

The original EM paper. This citation establishes the classical EM framework so that the notion of “implicit EM” has a clear reference point.

---

**Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.**

Provides textbook treatment of Gaussian mixture models, EM, and the collapse problem. In GMMs, the log-determinant term prevents singular solutions. The present work draws a direct analogy: InfoMax regularization plays the same stabilizing role for neural LSE objectives that the log-determinant plays in classical mixture models.

---

## 5. Decoder-Based Autoencoders and Historical Context

**Hinton, G. E., & Salakhutdinov, R. R. (2006). *Reducing the dimensionality of data with neural networks*. Science.**

Represents early motivation for unsupervised feature learning via layer-wise pretraining with RBMs. This work contextualizes why unsupervised objectives were historically important before being eclipsed by end-to-end supervised training.

---

**Vincent, P., et al. (2008). *Extracting and composing robust features with denoising autoencoders*. ICML.**

Established the encoder–decoder paradigm as the default unsupervised architecture. The present work identifies the resulting “decoder tax,” where reconstruction dominates the objective, and demonstrates that decoders are unnecessary when implicit EM and volume control are used instead.

---

## 6. Sparse Autoencoders and Interpretability

**Bricken, T., et al. (2023). *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning*. Anthropic.**

The landmark sparse autoencoder (SAE) paper for interpretability. Uses an encoder–decoder architecture with L1 sparsity. The present work does not claim empirical superiority, but provides theoretical grounding for why decoders may be unnecessary in principle.

---

**Cunningham, H., et al. (2023). *Sparse Autoencoders Find Highly Interpretable Features in Language Models*. arXiv:2309.08600.**

Demonstrates the practical utility of sparse features in large language models. Serves as empirical motivation for understanding and simplifying SAE objectives.

---

## 7. Activation Functions and Emergent Sparsity

**Glorot, X., Bordes, A., & Bengio, Y. (2011). *Deep Sparse Rectifier Neural Networks*. AISTATS.**

Shows that ReLU nonlinearities induce sparsity naturally. Relevant because the present architecture achieves sparsity without an explicit L1 penalty, relying instead on competition and rectification.

---

## 8. Optional: Variational Alternatives

**Kingma, D. P., & Welling, M. (2014). *Auto-Encoding Variational Bayes*. ICLR.**

VAEs represent an alternative probabilistic approach to representation learning that avoids pure reconstruction objectives. Cited as a contrasting framework rather than a direct precursor.

---

**Higgins, I., et al. (2017). *β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework*. ICLR.**

Demonstrates that stronger regularization leads to more disentangled representations. Parallels the present finding that strong InfoMax regularization is necessary for useful features, despite a different underlying mechanism.

---

## 9. Summary Table

| Citation | Primary Role |
|--------|--------------|
| Bell & Sejnowski (1995) | InfoMax principle |
| Linsker (1988) | Information preservation |
| Olshausen & Field (1996) | Sparse coding baseline |
| LeCun et al. (2006) | Energy-based framing |
| Kohonen (1982) | Competitive learning precursor |
| Oursland (2025) | Implicit EM identity |
| Oursland (2024) | Distance-based interpretation |
| Dempster et al. (1977) | Classical EM |
| Bishop (2006) | EM collapse analogy |
| Zbontar et al. (2021) | Decorrelation regularizer |
| Bardes et al. (2022) | Variance + covariance control |
| Glorot et al. (2011) | ReLU-induced sparsity |
| Bricken et al. (2023) | SAE interpretability context |

---
