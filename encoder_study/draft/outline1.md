Here is the detailed paper outline based on your files and our "fix in post" strategy for the distance/similarity inversion.

# Paper Outline: Deriving Decoder-Free Sparse Autoencoders from First Principles

## Abstract

Sparse Autoencoders (SAEs) are the standard tool for extracting interpretable features from neural networks, but they rely on a computationally expensive decoder and a heuristic reconstruction objective. We show that the decoder is unnecessary. By viewing neural activations as energy-based distances, we derive a decoder-free objective from the principles of **Implicit Expectation-Maximization (EM)**. We demonstrate that the standard Log-Sum-Exp (LSE) objective performs the E-step but suffers from dimensional collapse. We introduce **InfoMax regularization** (maximizing variance and minimizing correlation) as the necessary "volume control" analogous to the log-determinant in Gaussian Mixture Models. The resulting objective  allows a single linear layer to learn distinct, sparse features without reconstruction.

---

## 1. Introduction

* **The Problem:** Sparse autoencoders learn interpretable features but impose a "decoder tax"—doubling parameters and compute to reconstruct the input, raising the question of whether reconstruction is necessary or just a proxy for preventing feature collapse.
* **The Theoretical Gap:** Current unsupervised objectives often lack a unifying principle. We propose that gradient descent on specific objectives performs implicit EM, but requires regularization to replace the missing "volume" constraints found in mixture models.
* **Our Contribution:** We derive a decoder-free architecture from first principles. We show that LSE provides the mechanism (competition) and InfoMax provides the objective (structure), creating a stable unsupervised learner without reconstruction.

## 2. Theoretical Framework: Gradient Descent as Implicit EM

* **The Core Identity:** We begin with the observation that for Log-Sum-Exp objectives , the gradient with respect to energy is exactly the posterior responsibility: .
* **The Mechanism:** This proves that standard backpropagation performs the E-step (computing responsibilities) and the M-step (updating weights) simultaneously. No auxiliary variables are needed; the "responsibility" is implicit in the gradient flow.
* **The Collapse Problem:** In Gaussian Mixture Models, the log-determinant () prevents singularities. In neural LSE objectives, no such term exists. Without it, the objective admits degenerate solutions where one prototype covers all data or all prototypes collapse to the mean.

## 3. The Derivation: InfoMax as Volume Control

* **The Missing Term:** We introduce InfoMax regularization to serve the role of the log-determinant. It acts as a "volume control" that enforces existence and diversity.
* **Force 1: Attraction (LSE):** The term  minimizes the energy of the best-matching prototype. It pulls prototypes *toward* data to ensure coverage.
* **Force 2: Structure (InfoMax):**
* **Variance (Existence):** The term  penalizes dead units. It forces every prototype to have a non-zero "volume" of activity across the dataset.
* **Decorrelation (Diversity):** The term  penalizes redundancy, ensuring prototypes capture distinct factors of variation.


* **The Resulting Objective:** A min-max game where LSE distributes responsibility and InfoMax prevents collapse.

## 4. Architecture & Implementation (The "Fix")

* **The Polarity Shift:** We explicitly frame the encoder output  as a **distance/energy**, not similarity.
* **Encoder:** , .
* **Loss:**  minimizes distance (pulls  for matches).


* **Recovering Features:** For downstream tasks and benchmarking, we invert the distance back to similarity: . This transformation recovers the standard sparse vector (mostly zeros, few ones) from the distance vector (mostly large, few zeros).
* **Decoder-Free Design:** The model contains no decoder parameters. Reconstruction for analysis is performed using , but this is not required for training.

## 5. Experimental Validation

* **Experiment 1: Verifying the Theorem:** A scatter plot of analytical gradients vs. computed responsibilities ().
* *Prediction:* The points will lie on the  identity line, confirming the implicit EM dynamics.


* **Experiment 2: Ablation Study:** Comparing configurations of LSE, Variance, and Decorrelation.
* *Prediction:* LSE alone leads to collapse (one active unit). InfoMax alone leads to whitening without semantic meaning. The combination yields stable, diverse features.


* **Experiment 3: Benchmark Comparison:** Comparing the Decoder-Free model against a standard SAE (Encoder-Decoder + L1) on MNIST/LLM activations.
* *Metrics:* Sparsity (measured on ), Reconstruction MSE (via ), and Parameter Count.
* *Prediction:* Comparable reconstruction and sparsity with 50% fewer parameters.



## 6. Discussion & Prior Work

* **Connection to EBMs:** We frame our model as a mixture of energy-based components, where InfoMax regularizes the energy landscape.
* **Connection to Sparse Coding:** Our approach removes the reconstruction term from Olshausen & Field’s objective, relying on competitive allocation to induce sparsity.
* **Why It Matters:** This suggests that the "decoder" in SAEs was never about reconstruction; it was a mechanism to enforce non-triviality. With explicit InfoMax regularization, the encoder is sufficient.

## 7. Conclusion

* Decoder-free sparse autoencoders arise naturally from implicit EM when log-sum-exp objectives are combined with InfoMax volume control. The decoder is a redundant artifact of previous formulations.