# Scope: The Conditions Under Which Implicit EM Fails

This document describes the paper that the supervised study has become, and what remains to complete it. It supersedes the confirmation-framed supervised_study/draft/scope.md (which predates experiments 3–4).

## Pattern

Paper 1 derived the identity: gradients of LSE objectives are responsibilities. Paper 2 showed the theory is prescriptive in the unsupervised regime — and found an anomaly it did not predict: the model trained with SGD at any learning rate, Adam offered nothing, convergence time was fixed. This paper explains that anomaly, states precisely when it holds, and demonstrates the exact conditions under which it fails.

Paper 2 said: the theory can build a model. This paper says: here is the boundary of the theory, drawn from both sides.

## The Question

Implicit EM gives networks a distinctive optimization property — a landscape conditioned by construction, learning-rate insensitivity, fixed convergence time, no benefit from adaptive optimizers. Under what conditions does that property exist, and what destroys it?

## The Claim

The EM property is **local, graded, and structural**:

- **Local.** It exists exactly at sites of exponentiation + normalization, and extends zero layers beyond them. It cannot be inherited through a learned dense linear map.
- **Graded.** At a real site inside a network, the property holds in proportion to the share of the arriving gradient that is EM-structured. The transition between "behaves like single-layer EM" and "behaves like an ordinary MLP layer" is continuous in the gradient-dominance ratio.
- **Structural.** The failures divide into two classes with different remedies. *At-site* failures (collapse, dead components, redundancy) are repaired by volume control applied at the site — this transfers into supervised networks. *Structural* failures (loss of conditioning under composition) cannot be repaired by any penalty, because the decomposition EM consists of no longer exists — only isolation (stop-gradient) or dominance (λ) restores it.

And the punchline that ties it to sixty years of prior art: **backprop composition is parameter sharing in disguise**, and parameter sharing across components is the oldest known failure condition of classical EM (the Q-function stops separating). The neural failure is not new; it is the classical failure, rediscovered inside the chain rule.

## The Evidence (already in hand)

**Theory.**
- The invariant: the LSE gradient is a conditional expectation on the probability simplex (bounded, normalized, per-component aligned).
- The conditioning lemma (notes/conditioning_lemma.md): LSE curvature ≤ ½ uniformly (Böhning/Popoviciu); with private per-component parameters, parameter-space curvature ≤ ½·E‖x‖² — a data constant, invariant over training. Composed behind W₂, curvature contains W₂ᵀ(diag(p)−ppᵀ)W₂ and scales with σ_max(W₂)² — learned, growing, anisotropic.
- The operation classification (notes/why_composition_breaks_em.md): simplex-compatible operations (private parameters, per-coordinate monotone maps, stochastic maps) vs structure-destroying operations (shared sign-indefinite linear maps, objective mixing, hard gates), each mapped to a classical EM failure class.

**Experiments (supervised_study, reports 1–4).**
- Exp 1–2: volume control is needed at intermediate layers and transfers there (the at-site half). Partial volume control is worse than none.
- Exp 3: the conditioning does not survive composition (the structural half, observed).
- Exp 4: the mechanism, measured and causally established.
  - Gradient competition: CE outweighs the EM path 30–70× at λ=0.001, and the two are orthogonal (cos ≈ 0) — the EM signal is drowned, not opposed.
  - Curvature: LSE Hessian ≤ ½ at every point of training; CE-path Hessian 13–100× larger, non-monotone, parameter-dependent. Under pure EM training the ½ bound is *saturated exactly* (0.4993–0.4997) — the bound is a diagnostic of mixture sharpness, not just a worst case.
  - Causal test: cutting the CE→W₁ gradient path restores learning-rate insensitivity (probe-accuracy spread 10.9 pts → 1.3 pts across 1000× in lr) and fixed convergence time, inside a supervised network. λ=1 with full connectivity is identical to stop-gradient; λ=0.03 lands halfway. Dominance, not connectivity.
- The honest trade-off: EM-dominated features probe at ~88% vs ~96% for CE-dominated. Conditioning is bought with task alignment. The paper states this plainly.

**Corroborating observation (unpublished, Oursland).** EM-pretrained encoder fine-tuned by GD, and GD-trained autoencoder fine-tuned by the EM objective, both move enormous L2 distances in parameter space: the two objectives occupy different basins. This is the integrated version of the orthogonality measurement.

## Contributions

1. A provable characterization of when the EM optimization property holds (private parameters, monotone maps) and constants for when it fails (σ_max(W₂)² scaling).
2. The at-site / structural failure taxonomy, aligned with classical EM failure classes.
3. Causal experimental confirmation: the property is destroyed by gradient dominance through the composition corridor and restored by isolation or rebalancing.
4. The curvature-saturation diagnostic: distance of ‖∇²_d L_LSE‖ from ½ measures mixture sharpness at any site in any network — a probe usable beyond this paper.
5. A mechanistic account of why generative pretraining does not survive discriminative fine-tuning (dominant orthogonal gradient overwrites; nothing preserves the pretrained structure).

## What Is Still Required

Ranked. Items 1–4 are needed for submission; 5–7 strengthen it; 8 is the writing itself.

1. **Complete the Paper 2 signature under stop-gradient: the Adam arm.** Exp 4 swept SGD only. Run Adam × 4 lrs × 3 seeds in the stop-gradient and joint λ=0.001 arms. Prediction: Adam ≈ SGD under stop-gradient (no advantage), Adam ≫ SGD in the joint arm. Without this, "Paper 2's signatures reappear" covers two of three signatures. Cost: ~1 hour of CPU, zero new code (flag exists).

2. **The λ-resolution curve.** Three λ values sketch the transition; the claim "graded by gradient dominance" wants the full curve: 6–8 λ values, x-axis = *measured* CE/EM gradient ratio (not λ), y-axis = conditioning spread and probe accuracy. This is the paper's central figure. Cost: ~3 hours CPU, trivial code.

3. **Tighten the formal statements.** The lemma is proved for linear energies; the paper uses ReLU. Extend the proof to monotone kernels with bounded φ′, φ″ (Softplus — straightforward), state the ReLU case as a.e. bounds with the absorbing-state caveat, and either include LayerNorm in the constants or state the model without it. Replace the informal "iff" with what is actually proved: sufficient conditions for preservation, plus demonstrated failure under each violated condition. Cost: days of careful writing, no new mathematics.

4. **Basin analysis to formalize the fine-tuning observation.** Linear mode connectivity between the EM-arm and CE-arm solutions (checkpoints exist in results/experiment4): interpolate parameters, plot both losses along the path, with permutation alignment (Hungarian matching on hidden units) to kill the relabeling artifact. Raw L2 distance will not survive review; the barrier plot will. Cost: ~1 day including alignment code.

5. **Depth sweep (the "how far in" experiment).** Stack 2–6 EM layers, each with its own LSE + volume control site, train jointly and with per-layer stop-gradients; plot the ½-saturation diagnostic per layer. Tests whether local sites stay healthy under many corridors — the locality claim at depth, and the direct evidence for layer-wise EM as the design implication. Cost: ~2 days.

6. **The stochastic-map arm.** Replace W₂ with a non-negative row-normalized map. The one preservation class with no experiment. Prediction: partial conditioning survives. This is also the bridge to the attention-sink work in the related project — a positive result here predicts InfoMax-style control transfers to attention (at-site) while conditioning does not (structural). Cost: ~1 day; risk: the constraint may hurt head accuracy enough to confound the probe. Report either way.

7. **Robustness.** A second dataset (Fashion-MNIST is sufficient; the claims are about optimization structure, not vision), 5–10 seeds on headline results, one wider K. Cost: CPU time only.

8. **The writing.** New outline replacing outline.md (built when the confirmation framing was assumed). Positioning section: Xu & Jordan (EM as preconditioned gradient), Böhning (the bound), Dempster–Laird–Rubin (missing-information rate), Neal & Hinton (free-energy view), generative-vs-discriminative (Ng & Jordan), linear mode connectivity (Frankle et al.), and the layer-wise pretraining literature the mechanism reinterprets. Length target ~14 pages: more than Paper 2's scope because this paper carries both theory and experiments.

## What Is Out of Scope

- LLMs, transformers, attention experiments (one discussion paragraph connecting to the stochastic-map result; the experiments belong to the related project).
- SOTA or accuracy claims of any kind. The 88%-vs-96% trade-off is reported as a finding, not fought.
- A general theory of deep-network conditioning. Claims are about EM sites and their corridors, nothing more.
- Convergence proofs for the composed system.

## Risks

- **"This is just generative vs discriminative."** The defense is the mechanism: the paper does not observe that the objectives differ; it proves the curvature dichotomy, measures the orthogonality and dominance, and restores the property on command. Ng & Jordan is cited as the phenomenon; this is the account of it inside one architecture.
- **The iff temptation.** The preservation conditions are proved sufficient, and each violation is demonstrated empirically — but necessity in general is not proved. The writing must not claim it.
- **MNIST-only.** Mitigated by item 7 and by the nature of the claims (optimization structure, verified against exact constants like ½, not benchmark performance).

## The Contribution, In One Paragraph

Implicit EM gives a neural network a provably well-conditioned landscape — but only where its structural conditions hold: private parameters, monotone maps, an unmixed objective. We prove the conditioning, classify the operations that preserve and destroy it, show the destroying operations are the classical EM failure classes in neural form, and confirm causally that the property is lost in proportion to foreign gradient dominance and restored by isolation. Volume control transfers anywhere; conditioning transfers nowhere. Knowing which is which is what the theory now provides.
