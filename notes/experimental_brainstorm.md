## **Core Validation** (Must-haves)

1. **Feature Learning Quality**
   - Train on toy datasets with known ground truth features (e.g., sparse combinations of basis vectors, compositional data)
   - Does it recover the true features? How does reconstruction quality compare to standard SAEs?
   - Metrics: feature recovery, reconstruction loss, downstream task performance

2. **Standard Benchmark: Transformer Activations**
   - Train on MLP layer outputs from a small language model (GPT-2 small or similar)
   - Compare learned features to standard SAEs (vanilla, TopK, etc.) using:
     - Interpretability metrics (feature specificity, monosemanticity)
     - Sparsity levels achieved
     - Computational cost (parameters, FLOPs, memory)

3. **Ablations**
   - InfoMax alone vs LSE alone vs both
   - Different activation functions (ReLU, GELU, etc.)
   - Effect of batch size on responsibilities/learning dynamics

## **Unique Properties** (What makes this special?)

4. **Batch Dynamics**
   - How does batch size affect learning? (Since responsibilities are computed across batches)
   - What happens with different batch compositions?
   - Is there a "sweet spot" or does it scale gracefully?

5. **Feature Geometry**
   - How do learned features organize compared to standard SAEs?
   - Are there differences in feature correlation structure?
   - Does the responsibility mechanism lead to more/less feature splitting?

6. **Computational Scaling**
   - Parameter count vs feature quality curves
   - Training time/throughput comparisons
   - How does it scale to very wide feature spaces?

## **Theoretical Insights** (Connect to derivation)

7. **Connection to EM/Clustering**
   - Visualize responsibilities during training
   - Compare to explicit k-means or mixture model baselines
   - Can you show the implicit EM is actually happening?

8. **Information-Theoretic Analysis**
   - How does mutual information evolve during training?
   - Trade-offs between sparsity and information preservation
   - Does InfoMax alone explain the good behavior?

## **Nice-to-haves** (If space permits)

9. **Different Domains**
   - Vision (e.g., CIFAR-10 features)
   - Audio/speech
   - Shows generality beyond language models

10. **Downstream Applications**
    - Circuit analysis in small models
    - Steering/intervention experiments
    - Does the lack of decoder affect interpretability work?

**Paper Flow Suggestion:**
- Derivation → Toy experiment proving concept → Transformer benchmark → Ablations + unique properties → Discussion of when/why to use this

---

This is **beautiful** work! You've built a complete theoretical stack:

**Layer 1 (Mahalanobis paper)**: Linear layers compute distances to prototypes  
**Layer 2 (This paper)**: LSE objectives → ∂L/∂d_j = -r_j → implicit EM  
**Layer 3 (Decoder-free SAE)**: Linear → Activation → (LSE + InfoMax) → responsibilities replace decoder

The implicit EM paper is the **critical bridge**. It explains *why* LSE works for your decoder-free architecture - not just as a heuristic, but as a principled realization of EM where responsibilities emerge as gradients.

Now for the experiments paper, I'd suggest validating **all three layers** of the theory:

## Revised Experiment Priorities

**Must-haves (validate the theory stack):**

1. **Verify Implicit EM Dynamics** (validates Layer 2)
   - Extract responsibilities during training: compute softmax(z_i) per feature across batches
   - Show they behave like EM: components specialize, responsibilities sharpen over time
   - Compare to explicit EM baseline (classical GMM on same data)
   - This is *unique* to your approach - no other SAE paper has this connection

2. **Standard SAE Benchmark** (practical validation)
   - Transformer activations (GPT-2 small)
   - Compare to vanilla SAE, TopK, Gated SAE
   - Metrics: reconstruction, sparsity, interpretability, compute

3. **Ablation: LSE vs InfoMax** (isolate components)
   - LSE alone, InfoMax alone, both
   - Show they're complementary - LSE gives EM structure, InfoMax prevents collapse

**Should-haves (mechanistic insights):**

4. **Batch Size Effects** (unique to cross-batch responsibilities)
   - How does batch size affect learning dynamics?
   - Does it mirror mini-batch EM literature?

5. **Feature Geometry Analysis** (validates Layer 1 - distance interpretation)
   - Are learned features actually distance-based (Mahalanobis-like)?
   - Compare feature correlations to standard SAEs
   - Visualize "prototype" structure

6. **Toy Problem with Ground Truth** (cleanest validation)
   - Synthetic data: known sparse generative model
   - Does decoder-free recover true features?
   - Compare responsibility evolution to theoretical EM

**Nice-to-haves:**

7. **Computational Scaling** (practical advantage)
   - Parameters saved, training throughput, memory
   
8. **Downstream Applications** (circuit analysis, steering)

The **story** of the paper should be:

> "We derive decoder-free SAEs from first principles by recognizing that (1) linear layers compute distances [prior work], (2) LSE objectives perform implicit EM via responsibility gradients [this paper establishes], therefore (3) LSE+InfoMax eliminates the decoder by replacing explicit reconstruction with implicit EM. Experiments validate the theoretical predictions: features exhibit EM specialization dynamics, match or exceed standard SAEs, and scale more efficiently."

The implicit EM paper gives you a **unique theoretical hook** that no other SAE architecture has. The experiments should showcase this!

---

This is actually **more powerful** than traditional GMMs because:

**Traditional GMM:** Small number of complete multivariate Gaussians → forces data into discrete blobs

**Your "Gaussian Component Field":** Many scattered univariate Gaussian directions → can capture complex manifold geometry

Each feature learns:
- A direction in space (the weight vector)
- A prototype location (the bias)
- A scale (implicitly learned)

The field is "spaghettified" because:
- Directions can overlap, cross, interfere
- Not constrained to be orthogonal or organized into neat clusters
- Can adapt to the tangled structure of high-dimensional data
- More like a neural representation of a manifold than a mixture model

**This changes the experiment focus entirely:**

## Revised Experiments for "Gaussian Component Field"

**Validate the field structure:**

1. **Visualize the Field** (most important!)
   - Project learned weight vectors - are they scattered/tangled?
   - Show they're NOT organized into orthogonal sets
   - Demonstrate coverage of the manifold
   - Compare to PCA/ICA (which force orthogonality)

2. **Feature Specialization Patterns**
   - Do features capture *local* directions on a manifold rather than global clusters?
   - Show individual features respond to specific input variations
   - Demonstrate the field adapts to data geometry

3. **Reconstruction from Field**
   - Even without a decoder, can you reconstruct by summing feature contributions?
   - Does the scattered field provide better coverage than discrete Gaussians?

4. **Compare to Structured Alternatives**
   - vs GMM (full covariance): too rigid, too few components
   - vs Factor Analysis: forces hierarchical structure
   - vs PCA/ICA: forces orthogonality
   - Show your field is MORE flexible

5. **Manifold Coverage Metrics**
   - Does the field uniformly cover the data manifold?
   - Gaps in coverage? Redundant regions?
   - Compare to alternative sparse coding methods

**The story becomes:**

> "Sparse Autoencoders attempt to decompose data into interpretable features. Traditional approaches use decoder reconstruction. But reconstruction forces features into a specific linear combination structure. We derive a decoder-free architecture that instead learns a *field* of Gaussian directions through implicit EM. Each feature captures one direction of manifold variation. The resulting field is scattered, adaptive, and captures complex geometry that structured models cannot."

---

If you want this paper to land, the experiments should do three jobs:

1. prove the *mechanism* (implicit responsibilities + competitive specialization),
2. prove the *object* you claim you learned (a “Gaussian Component Field,” not a GMM),
3. prove it’s *useful* (representations / interpretability / downstream or compression).

Here’s the experimental package I’d run.

Start with “does it do what the derivation says?”
**A. Responsibility dynamics are EM-like**

* Plot responsibility entropy per sample over training: (H(r(x))) should drop (competition sharpens) but not collapse to a single component globally.
* Show “two-timescale” behavior if it exists: responsibilities stabilize earlier than weights (or vice versa), like E/M separation.
* Track effective number of active components:

  * global usage: (N_{\text{eff}} = \exp(H(\mathbb{E}_x[r(x)])))
  * per-sample usage: (\exp(H(r(x))))
    This is the cleanest way to demonstrate “field coverage” vs “one component wins.”

**B. Specialization**

* For each unit/component, collect top-activating samples and show they’re coherent (images/text patches).
* Show that each unit wins responsibility in distinct regions: heatmap of component usage across data clusters.

Then prove “field not Gaussians”
**C. Fieldness vs GMM-ness on synthetic data**
Use 2D and 3D toy distributions where the “right” answer differs:

1. True GMM with a few ellipses (should be easy for a GMM; does your model recover components or still produce strands?)
2. Curved manifold (e.g., two moons, Swiss roll) where a GMM struggles but a field should tile locally.
3. Union of subspaces (mixture of lines/planes) where “principal-direction factors” are the natural object.

For each, show:

* Learned “directions” (rows of (W)) as vectors anchored in input space (2D/3D plots).
* Responsibilities as a soft tessellation of space.
* Compare against:

  * GMM (full covariance) fit
  * k-means
  * ICA (if you want the factor story)
  * (optional) a decoder SAE trained on same data.

Key metric here isn’t just log-likelihood; it’s **geometric coverage**:

* nearest-component energy over space,
* responsibility partition smoothness,
* how many components are needed to cover manifold at fixed error.

Then show collapse and why your regularizer exists
**D. Ablation grid that reproduces your “failure modes” table**
This is essential because it validates the derivation logic.

Run:

* LSE only → collapse
* * variance only → no dead units but redundancy
* * corr only → fewer duplicates but dead units possible
* * variance + corr → stable but weight loopholes appear
* * all (including (W^TW)) → stable and diverse

For each condition report:

* global usage entropy (H(\mathbb{E}[r]))
* fraction of dead units (low variance or near-zero usage)
* redundancy score (off-diagonal corr norm)
* “scale blow-up” indicator (mean/var of activations over training)

This section should basically write itself from your theory.

Then show it learns something nontrivial
**E. Representation quality (decoder-free)**
Pick one of these lanes (or do two if you can afford it):

1. **Linear probe**: freeze encoder, train linear classifier on CIFAR-10 / MNIST / FashionMNIST / tiny ImageNet subset. Compare to:

   * random features
   * same encoder trained with SimCLR-ish baseline (if you want)
   * Barlow Twins / VICReg (closest philosophical baseline: variance + covariance).
2. **Clustering quality**: kNN accuracy in representation space, or NMI/ARI with labels.

Also report **sparsity** in a way that matches your mechanism:

* distribution of (H(r(x))) (responsibility sparsity)
* optionally, activation sparsity (% zero for ReLU), but don’t oversell it—your real sparsity is in (r).

Then show interpretability (this is where the paper can be memorable)
**F. “Component Field” interpretability artifacts**
For vision:

* For each unit: top-k image patches or feature visualizations.
* Show spatial maps if you do conv features (where does each factor fire?).
* Show that units correspond to local principal directions: perturb input along (w_j) direction and show predictable change in activation/responsibility.

For language:

* For each unit: top contexts/tokens where it wins; show they cluster semantically.

Then a very compelling “sanity” experiment
**G. Out-of-distribution / open set behavior**
Your implicit-EM paper highlights the closed-world softmax issue. Your model is *mixture-like*, so it might do something interesting:

* Evaluate on OOD (e.g., MNIST trained, test on FashionMNIST; CIFAR-10 vs SVHN).
  Report:
* max responsibility (confidence proxy)
* energy minimum (\min_j E_j(x))
* entropy (H(r(x)))

Field models often show higher entropy or higher min-energy on OOD—if you see that, it’s a strong story.

Finally, if you want to connect back to the Mahalanobis angle explicitly
**H. “Mahalanobis-ness” check**
On data with known covariance structure (synthetic Gaussian clusters), test whether:

* learned (W) rows align with principal directions (cosine similarity to true eigenvectors),
* orthogonality regularizer makes this alignment sharper (ties back to the whitening non-uniqueness discussion).

What I’d consider “minimum viable” for the paper
If you want the shortest set that still convinces:

1. Synthetic geometry demo showing field behavior vs GMM
2. Ablations proving collapse and the role of your regularizers
3. One real dataset with linear probe + interpretability gallery
4. Responsibility/usage metrics over training

A note on baselines (choose a few, don’t boil the ocean)
The most honest comparators are:

* GMM (full cov) and k-means (classic mixture baselines)
* Barlow Twins or VICReg (variance + covariance regularization cousins)
* A standard decoder SAE (to show what you get/lose by dropping decoder)
* (optional) VQ-VAE / k-means quantizer if you want the “soft VQ” comparison

If you tell me what modality you’re targeting first (images vs language) and what scale (MNIST/CIFAR vs bigger), I’ll turn the above into a tight experimental section outline with exact plots, metrics, and expected qualitative outcomes.

---

To turn your "Trilogy" into a defensible paper, you need to empirically prove that your **Decoder-Free** architecture actually works as well as (or better than) the standard **Decoder-Heavy** SAEs used by labs like Anthropic.

Since your claim is that **Optimization (LSE) = Inference (EM)** and **InfoMax = Volume Control**, your experiments must validate these specific mechanisms.

Here is a recommended experimental roadmap for *"Deriving Decoder-Free Sparse Autoencoders from First Principles"*:

### 1. The "Decoder-Free" Validation (The Reconstruction Test)

The biggest skepticism you will face is: *"If you don't have a decoder, how do you know you kept the information?"*
Your theory implies that if the encoder  learns the true generative directions (principal components), then the transpose  is the optimal decoder naturally.

* **Experiment:** Train your model on a standard dataset (e.g., GPT-2 small residuals or MNIST/CIFAR patches).
* **The Check:** Periodically measure the **Reconstruction Error** () using the *frozen, transposed* encoder weights as the decoder.
* **Hypothesis:** If your implicit EM derivation works, the reconstruction error using  should decrease and converge to a baseline comparable to a standard Tied-Weight Autoencoder, *even though you never explicitly minimized reconstruction error*.
* **Why this matters:** It proves the "Decoder-Free" claim isn't just throwing away data; it’s compressing it into a reversible basis.

### 2. The "Volume Control" Ablation (The InfoMax Test)

You claimed InfoMax prevents the "collapse" predicted in your 2025 paper (Section 6.2). You need to demonstrate this failure mode.

* **Experiment:** Train three variants:
1. **Pure LSE:** No InfoMax term.
2. **LSE + Weight Decay:** The "heuristic" volume control you mentioned in the paper.
3. **LSE + InfoMax:** Your proposed solution.


* **Metrics to Plot:**
* **Dead Neuron Count:** Percentage of neurons with 0 responsibility.
* **Feature Collapse:** The average cosine similarity between feature vectors in .


* **Visual:**  comparing the three.
* *Hypothesis:* Pure LSE will show high similarity (collapse to mean) or many dead neurons. InfoMax should show a clean, diagonal-heavy similarity matrix (orthogonal-ish "Component Field").



### 3. The "Component Field" vs. "Cluster" Analysis

To distinguish your work from clustering (K-Means) and prove it creates a "Component Field" (ICA-like), you need to analyze co-occurrence.

* **Experiment:** Analyze the distribution of active features per input.
* **The Check:**
* **Clustering (GMM/K-Means):** Usually 1 feature active (Winner-Take-All).
* **Component Field (Yours):** A sparse *set* of features active (Winner-Take-Most).


* **Visualization:** Plot a histogram of "Number of Active Features" (L0 norm) per input.
* *Goal:* You want to see a Poisson-like distribution (e.g., 5-10 features active), not a binary "1 or 0" distribution. This confirms you are learning disentangled factors, not just centroids.



### 4. Head-to-Head vs. Standard SAE

You need a benchmark. Compare your model against a standard **ReLU SAE** or **TopK SAE** (using the same number of parameters, but counting the decoder parameters against them).

* **Metric:** **Linear Probe Accuracy**.
* Freeze the latent codes  generated by your model and the baseline SAE.
* Train a simple linear classifier on top of  to solve a downstream task (e.g., predicting next token class or image label).


* **Hypothesis:** Your model achieves similar downstream accuracy with **half the memory footprint** (since you don't store a decoder).

### 5. Verify the "Gradient = Responsibility" Theorem

Since this is a "First Principles" paper, you should visually validate the core theorem from your 2025 paper.

* **Experiment:** During a single training step, record:
1. The calculated **Responsibility** .
2. The actual **Gradient**  flowing back from your LSE objective.


* **Visual:** A scatter plot of  vs. .
* *Result:* It should be a perfect  line (identity). This is the "mic drop" chart that proves your math is implemented correctly and the "Implicit EM" is real.



### Summary of Recommended Figures

| Figure | Goal | Tag Suggestion |
| --- | --- | --- |
| **1. The Collapse** | Show that without InfoMax,  degenerates (all features look the same). |  |
| **2. The Implicit Decoder** | Show  reconstructs  effectively without explicit training. |  |
| **3. Feature Interpretabilty** | Show "Component Field" features (e.g., specific curve detectors or semantic text concepts). |  |
| **4. The Identity** | Scatter plot proving Gradient == Responsibility. |  |

**Do you have a preference for Image/Vision datasets or Text/LLM residuals for these experiments?** (Text is the hot topic for SAEs right now, but Vision makes "Component Fields" easier to visualize).

---

