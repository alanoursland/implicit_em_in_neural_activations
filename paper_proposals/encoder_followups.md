# Proposal: Follow-ups to the Decoder-Free Encoder Paper

Parent paper (published): *Deriving Decoder-Free Sparse Autoencoders from First Principles*, Oursland, arXiv:2601.06478 (2026-01-10). Source study: encoder_study/.

The published paper's own future-directions section lists nine threads. They are not nine papers. This document sorts them: which are absorbed by other proposals, which are discussion paragraphs, and which two are standalone papers with a real path to completion.

## Standalone paper A: LSE as a Free Anomaly Detector

**Thesis.** High LSE loss means no mixture component explains the input. Every trained decoder-free encoder therefore ships with an anomaly score at zero additional cost — no reconstruction pathway, no density model, no modification. The paper benchmarks this score against reconstruction error from standard autoencoders and SAE baselines on held-out-class and corrupted-input protocols (MNIST/Fashion-MNIST/CIFAR held-out classes; corruption suites), and characterizes *which failure modes each catches*: reconstruction error fires on pixel-level novelty, LSE fires on "no prototype claims this" — predictably different errors, e.g. a novel combination of familiar strokes should fool reconstruction and trigger LSE.

**Evidence in hand.** The trained models and training code exist; the score is one line (the loss itself). Paper 2 established the components are interpretable prototypes, which is what gives the score its semantics.

**Required.** Benchmark harness (~2 days), baselines (standard AE, SAE, deep-SVDD-style one-class baseline; ~2 days), the held-out and corruption protocols across 3 datasets (~CPU days), calibration analysis (does the responsibility entropy add signal beyond the LSE value?), writing. The one theory piece worth adding: the closed-world caveat from notes/open_world_and_rejection.md — LSE is still a *relative* score over a closed hypothesis set; state what that implies about its failure cases (an anomaly near one prototype direction scores as normal). Honest scoping there is what separates this from a metrics-chasing paper.

**Risk.** Anomaly detection benchmarks are crowded. The differentiator is not SOTA but the free-ness and the failure-mode dichotomy; if LSE does not beat reconstruction error anywhere, the failure-mode characterization is still publishable, but as a smaller paper.

## Standalone paper B: Explicit EM for the Decoder-Free Model

**Thesis.** Paper 2 trains by gradient descent on an objective whose gradients *are* responsibilities. Take the last step: derive the closed-form M-step for the linear-energy case and train by actual EM — no learning rate, no optimizer. The failure-conditions work proved the GD landscape has fixed curvature ≤ ½·E‖x‖² and fixed convergence time; explicit EM should match or beat it and comes with monotone-improvement guarantees. Compare convergence (wall-clock and iterations), final features, and basins (are the EM and GD solutions the same solution? — connects to the measured basin separation between EM-style and task-driven training).

**Required.** The M-step derivation with the InfoMax terms included is the research content: LSE alone gives a softmin-weighted least-squares update (closed form); variance and decorrelation couple across the batch and likely need a generalized-EM inner step — proving a monotone GEM scheme that includes them is the theorem. Implementation and comparison are ~1 week. If the InfoMax-inclusive M-step resists closed form, the honest fallback is EM for LSE + gradient steps for InfoMax (an EM/GD hybrid), which is still a result.

## Absorbed elsewhere (do not duplicate)

- **Supervised regularization, conditioning of pretrained models, fine-tuning basin analysis** → failure_conditions.md (the basin experiments are item 4 there; the L2 pilot results already exist).
- **Layer-wise pretraining, hierarchical mixtures** → layerwise_em.md.
- **Attention volume control, head decorrelation** → attention_em.md.
- **Activation geometry (ReLU vs Softplus prototype character)** → one section of the failure-conditions paper (hard gates) plus the infomax_activations sweep; not standalone.

## Discussion paragraphs, not papers (for now)

- **Optimal transport connection.** Speculative until someone exhibits the transport cost; revisit if a derivation appears.
- **Post-hoc interpretability of pretrained models.** Real but unbounded scope; becomes concrete only with a specific model family and question — currently that concreteness lives in the related attention-sink project, not here.
