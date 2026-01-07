Here is the curated list of applicable citations, combining your seed list with critical references identified in the provided documents (`infomax_as_em_objective.md`, `infomax_vs_pretraining_prior_art.md`).

### **Core Theory (The "First Principles")**

* **Oursland, A. (2025). Gradient Descent as Implicit EM in Distance-Based Neural Models. arXiv:2512.24780.**
* **Applicability:** This is the primary theoretical foundation. It establishes the mathematical identity  that allows standard backpropagation to be interpreted as the E-step and M-step of an EM algorithm.


* **Oursland, A. (2024). Interpreting Neural Networks through Mahalanobis Distance. arXiv:2410.19352.**
* **Applicability:** Provides the geometric interpretation of linear layers as computing distances to prototypes. It justifies the "distance-based" view of activations () and predicts the necessity of the weight orthogonality term used in the final objective.


* **LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M., & Huang, F. J. (2006). A tutorial on energy-based learning. In Predicting Structured Data.**
* **Applicability:** Establishes the framework of Energy-Based Models (EBMs). Your LSE objective is identified as the negative log marginal likelihood under a mixture of energy functions defined by this framework.



### **The Objective (InfoMax & Regularization)**

* **Bell, A. J., & Sejnowski, T. J. (1995). An information-maximization approach to blind separation and blind deconvolution. Neural Computation.**
* **Applicability:** The primary inspiration for the "InfoMax" regularization terms. It provides the theoretical basis for maximizing variance and minimizing correlation to achieve independent features.


* **Linsker, R. (1988). Self-organization in a perceptual network. Computer.**
* **Applicability:** seminal work on the principle of maximum information preservation in neural layers. Cited to credit the broader "InfoMax" philosophy used to justify the volume control terms.


* **Zbontar, J., et al. (2021). Barlow Twins: Self-Supervised Learning via Redundancy Reduction. ICML.**
* **Applicability:** Specifically referenced as the source for the "Decorrelation Proxy" loss term (). It provides the tractable implementation for the redundancy reduction part of your InfoMax objective.



### **Architecture & Dynamics**

* **Olshausen, B. A., & Field, D. J. (1996). Emergence of simple-cell receptive field properties by learning a sparse code for natural images. Nature.**
* **Applicability:** Defines the standard "Sparse Coding" objective (Reconstruction + Sparsity) that this paper aims to supersede. It serves as the baseline model from which you remove the decoder.


* **Kohonen, T. (1982). Self-organized formation of topologically correct feature maps. Biological Cybernetics.**
* **Applicability:** Connects your implicit EM dynamics to Competitive Learning. It frames the "push-pull" equilibrium of your model (LSE vs. InfoMax) as a differentiable, probabilistic version of Self-Organizing Maps.



### **Historical Context (Why "Decoder-Free" Matters)**

* **Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the dimensionality of data with neural networks. Science.**
* **Applicability:** Contextualizes the work within the history of layer-wise pretraining (RBMs). It highlights the original motivation for unsupervised feature learning that was later abandoned, which this paper seeks to revive with a better objective.


* **Vincent, P., et al. (2008). Extracting and composing robust features with denoising autoencoders. ICML.**
* **Applicability:** Represents the "Autoencoder Era" that established the Encoder-Decoder architecture as the default. Cited to define the "Decoder Tax" problem (reconstruction focus) that your method eliminates.

---

That instinct is probably right, and that’s okay — arXiv is actually the *correct* home for a paper whose main contribution is conceptual unification rather than a leaderboard result.

Below is a clean, expandable citation list. Each entry has a short paragraph explaining *exactly* why it belongs, not just that it’s related. I’ve kept these tight and defensible, the way you’d want them in a Related Work or Discussion section.

---

**Bell, A. J., & Sejnowski, T. J. (1995). *An information-maximization approach to blind separation and blind deconvolution*. Neural Computation.**
This work introduced InfoMax as a principle for learning independent components by maximizing output entropy and minimizing redundancy. The present paper borrows the *operational* insight rather than the exact mutual information formulation: variance maximization and decorrelation serve as practical proxies for independence. InfoMax here plays the role of volume control in implicit EM, rather than serving as a complete statistical objective.

---

**Kohonen, T. (1982). *Self-organized formation of topologically correct feature maps*. Biological Cybernetics.**
Kohonen’s competitive learning dynamics show how prototypes distribute themselves to cover input space through attraction to data and mutual competition. The model derived in this paper recovers similar dynamics—prototypes specialize in different regions—but does so in a fully differentiable, probabilistic framework with soft responsibilities arising from implicit EM rather than hard winner-take-all updates.

---

**LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M., & Huang, F. J. (2006). *A tutorial on energy-based learning*.**
This tutorial formalizes the idea of defining learning objectives via energy functions rather than explicit probabilistic models. The log-sum-exp term in this paper is precisely the negative log marginal likelihood of a mixture of energy-based components. The InfoMax regularization can be interpreted as shaping the energy landscape to prevent degenerate minima, placing this work squarely within the energy-based modeling tradition.

---

**Linsker, R. (1988). *Self-organization in a perceptual network*. Computer.**
Linsker showed that maximizing information transmission alone can lead to the spontaneous emergence of structured, meaningful representations. This paper follows the same philosophical stance: useful features arise from information-theoretic constraints rather than task supervision. The present work extends this idea by embedding InfoMax inside an implicit EM mechanism, giving structure to the learning dynamics rather than relying on unconstrained gradient ascent.

---

**Olshausen, B. A., & Field, D. J. (1996). *Emergence of simple-cell receptive field properties by learning a sparse code for natural images*. Nature.**
Sparse coding demonstrated that interpretable features can emerge from unsupervised learning when sparsity is enforced via reconstruction and an L1 penalty. The present work reaches similar outcomes—sparse, interpretable features—but removes the decoder entirely. Sparsity here emerges from competitive responsibility allocation rather than explicit sparsity penalties, reframing sparse coding as an implicit EM problem with volume control.

---

**Oursland, A. (2024). *Interpreting Neural Networks through Mahalanobis Distance*. arXiv:2410.19352.**
This work provides the geometric interpretation that linear layers compute distances to prototypes under a Mahalanobis metric. That interpretation motivates treating encoder activations as energies and weight vectors as prototype directions. It also predicts the need for orthogonality or redundancy control in the weights, which reappears here as part of the full volume-control regularization.

---

**Oursland, A. (2025). *Gradient Descent as Implicit EM in Distance-Based Neural Models*. arXiv:2512.24780.**
This paper establishes the exact algebraic identity showing that gradient descent on log-sum-exp objectives performs expectation-maximization implicitly. The present work builds directly on that result, supplying the missing objective-side constraints—via InfoMax—that make implicit EM viable for unsupervised representation learning without collapse.

---

# Applicable Citations

## Foundational: InfoMax and ICA

**Bell, A. J., & Sejnowski, T. J. (1995). An information-maximization approach to blind separation and blind deconvolution. Neural Computation.**

Established InfoMax as a principle for learning independent components. Our variance and decorrelation terms are operational approximations of this principle. We cite this as inspiration for the regularization approach, while being careful not to claim we compute mutual information exactly.

**Linsker, R. (1988). Self-organization in a perceptual network. Computer.**

Original formulation of the InfoMax principle for neural networks. Linsker showed that maximizing information transmission leads to useful representations. We invoke this principle to justify our variance term (entropy proxy) as volume control.

---

## Foundational: Sparse Coding

**Olshausen, B. A., & Field, D. J. (1996). Emergence of simple-cell receptive field properties by learning a sparse code for natural images. Nature.**

The original sparse coding paper. Uses reconstruction + L1 sparsity. We position our work as removing the reconstruction term—sparsity emerges from competition rather than explicit penalty. This is the baseline conceptual approach we're simplifying.

---

## Foundational: Energy-Based Models

**LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M., & Huang, F. J. (2006). A tutorial on energy-based learning. In Predicting Structured Data.**

Comprehensive treatment of energy-based models. Our LSE term is the negative log marginal likelihood under a mixture of energy functions. We frame our encoder outputs as energies, and this tutorial provides the conceptual vocabulary for that interpretation.

---

## Foundational: Competitive Learning

**Kohonen, T. (1982). Self-organized formation of topologically correct feature maps. Biological Cybernetics.**

Self-organizing maps use competitive learning where prototypes compete to represent inputs. Our model has similar dynamics: LSE creates soft competition, InfoMax prevents collapse. We cite this as a non-probabilistic precursor with hard assignments where we use soft responsibilities.

---

## Foundational: This Work's Theoretical Basis

**Oursland, A. (2025). Gradient Descent as Implicit EM in Distance-Based Neural Models. arXiv:2512.24780.**

Establishes the core identity: for LSE objectives, gradient equals responsibility. This paper provides the implicit EM framework we build upon. We extend it by identifying InfoMax as the missing volume control that makes unsupervised implicit EM practical.

**Oursland, A. (2024). Interpreting Neural Networks through Mahalanobis Distance. arXiv:2410.19352.**

Interprets linear layers as computing distances to prototypes. Motivates viewing encoder outputs as energies/distances rather than similarities. Also predicts the need for orthogonality regularization on weights—which we include as optional λ_wr term.

---

## Sparse Autoencoders

**Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly, T., ... & Olah, C. (2023). Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. Anthropic.**

The landmark paper establishing SAEs for LLM interpretability. Uses standard encoder-decoder architecture with L1 sparsity. We position our work as providing theoretical grounding for why decoders might be unnecessary, though we don't claim to outperform their empirical results.

**Cunningham, H., Ewart, A., Riggs, L., Huben, R., & Sharkey, L. (2023). Sparse Autoencoders Find Highly Interpretable Features in Language Models. arXiv:2309.08600.**

Another key SAE paper for interpretability. Demonstrates practical utility of sparse features. Relevant as empirical motivation for why understanding SAE objectives matters.

---

## EM Algorithm and Mixture Models

**Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. Journal of the Royal Statistical Society: Series B.**

The original EM paper. We claim gradient descent performs EM implicitly for LSE objectives. This citation establishes what classical EM is, so readers understand what "implicit EM" means.

**Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.**

Textbook treatment of GMMs, EM, and the collapse problem. Section on GMMs explains the role of the log-determinant in preventing singularities. We draw the analogy: InfoMax is to neural LSE as log-determinant is to GMMs.

---

## Self-Supervised Learning with Similar Regularizers

**Zbontar, J., Jing, L., Misra, I., LeCun, Y., & Deny, S. (2021). Barlow Twins: Self-Supervised Learning via Redundancy Reduction. ICML.**

Uses cross-correlation loss to prevent redundancy in self-supervised learning. Our decorrelation term ||Corr(A) - I||² is essentially the same penalty. Different context (contrastive learning vs. sparse coding), but same insight: decorrelation prevents collapse.

**Bardes, A., Ponce, J., & LeCun, Y. (2022). VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning. ICLR.**

Explicitly decomposes self-supervised objectives into variance, invariance, and covariance terms. Our variance and decorrelation terms match their variance and covariance terms exactly. Strong evidence that these regularizers are broadly useful for preventing collapse.

---

## Neural Network Interpretability

**Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., ... & Olah, C. (2022). A Mathematical Framework for Transformer Circuits. Anthropic.**

Establishes the circuits research agenda that motivates SAE work. Understanding why SAEs work matters because they're tools for this research program. Provides application context.

**Olah, C., Mordvintsev, A., & Schubert, L. (2017). Feature Visualization. Distill.**

Early work on interpreting neural network features. Establishes that understanding learned representations is valuable. Motivates why a principled SAE derivation matters.

---

## Activation Functions and Sparsity

**Glorot, X., Bordes, A., & Bengio, Y. (2011). Deep Sparse Rectifier Neural Networks. AISTATS.**

Shows ReLU induces sparsity naturally. Relevant because our architecture uses ReLU and achieves sparsity without explicit L1 penalty. The sparsity emerges from competition + ReLU, not from a sparsity loss term.

---

## Optional: Variational and Information-Theoretic Approaches

**Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes. ICLR.**

VAEs use probabilistic encoding with KL regularization. Different approach to the same problem (learning useful latent representations). We could cite as alternative framework that also avoids pure reconstruction, though our approach is non-variational.

**Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick, M., ... & Lerchner, A. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. ICLR.**

Shows that stronger regularization (higher β) leads to more disentangled representations. Parallels our finding that InfoMax regularization is necessary for good features. Different mechanism, similar insight about regularization strength.

---

## Summary Table

| Citation | Why Cited |
|----------|-----------|
| Bell & Sejnowski 1995 | InfoMax principle; our regularization inspiration |
| Linsker 1988 | Original InfoMax formulation |
| Olshausen & Field 1996 | Sparse coding baseline we simplify |
| LeCun et al. 2006 | Energy-based model framing |
| Kohonen 1982 | Competitive learning precursor |
| Oursland 2025 | Implicit EM identity we build on |
| Oursland 2024 | Distance interpretation of linear layers |
| Bricken et al. 2023 | SAE for interpretability (application) |
| Dempster et al. 1977 | Classical EM definition |
| Bishop 2006 | GMM collapse and log-determinant |
| Zbontar et al. 2021 | Barlow Twins; same decorrelation term |
| Bardes et al. 2022 | VICReg; same variance + covariance terms |
| Glorot et al. 2011 | ReLU sparsity without L1 |