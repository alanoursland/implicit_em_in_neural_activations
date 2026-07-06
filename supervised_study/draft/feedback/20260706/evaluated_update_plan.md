# Evaluated Update Plan

This file consolidates the section-by-section feedback in this directory, filters it against the current draft, and turns it into an update list by target file.

Priority labels:

- **P0**: correctness, broken promise, or claim/body mismatch.
- **P1**: likely reviewer objection or important clarity fix.
- **P2**: polish, optional strengthening, or citation hygiene.

## Critical Evaluation of the Feedback

The feedback is mostly strong. It correctly identifies the paper's main remaining risks: a quantitative promise in the theory that is not cashed in the results, overstrong abstract/conclusion wording relative to the limitations, repeated future-work blocks, and a few places where empirical diagnostics are used more strongly than their measurement protocol supports.

The most important valid points are:

- The `E||xtilde||^2` descent-lemma comparison is explicitly promised in `theory.tex` and currently not delivered elsewhere.
- The `30--70x` CE/EM dominance number is measured under Adam while the causal conditioning sweep uses SGD; the text now discloses this, but the abstract and mechanism narrative still rely heavily on it.
- The single-layer optimizer signature is imported from prior work rather than shown side-by-side in the same units, which weakens the "disappears/returns" narrative. This is a persuasiveness issue, not a correctness defect, as long as the prior work is cited clearly.
- The future-strengthening material is repeated in setup, results, and limitations; this reads like unfinished work rather than controlled scope.
- Several claims should be scoped to "controlled MNIST evidence supports/bears out" rather than "confirms."
- Metrics in Results I partly overlap with the training losses, so the paper should acknowledge metric-objective coupling.
- Results II/III learning-rate sensitivity mostly shows low-learning-rate slowness, not an upper stability edge; this should be either tested or stated more carefully.

Some feedback is directionally useful but should not be applied literally:

- The cosine values being much smaller than `1/sqrt(K)` should not be upgraded to a strong "more orthogonal than chance" claim without a better null model. Batch averaging and shared structure can change the relevant null. A cautious note is better.
- "Run all cheap experiments" is not always an update to the manuscript. Where new runs are not available, the text should either soften the claim or move the item to limitations.
- Citation suggestions should be verified before being added. Do not bulk-add uncertain references just because they were suggested.
- The `1/2`-saturation diagnostic is already defined in `notation.tex`; the remaining issue is local discoverability near the lemma/table, not total absence.
- The theory/proof math has already been substantially repaired. Remaining proof feedback is mainly presentation/scope and verification-script quality.

## Self-Check on This Evaluation

My evaluation could still be too conservative in two places. First, matched-SGD gradient ratios and Adam-under-stop-gradient are likely cheap and would materially strengthen the paper; if those logs/runs are available, prefer adding them over hedging. Second, the single-layer side-by-side comparison may be the most persuasive fix for the whole paper, even if it is not a P0 requirement because the claim can be supported by citation to prior work.

My evaluation could also be too aggressive in treating the `E||xtilde||^2` promise as P0. If the descent-lemma comparison is not central to the final story, deleting the promise is an acceptable resolution. What is not acceptable is leaving it promised and uncited by results.

## Cross-Cutting Decisions

1. **Decide the `E||xtilde||^2` path.** Either compute/report the comparison, or remove the promise from `theory.tex`.
2. **Inventory available data before wording around it.** Check whether existing logs already contain matched-SGD CE/EM ratios, Adam stop-gradient, one larger SGD learning-rate rung, baseline optimizer sweep, single-layer signature numbers, capacity-table SDs, K=64 lambda recalibration, or probe variants. If data exist, prefer reporting them; if not, soften or move to limitations.
3. **Decide whether new runs are in scope.** Highest value if not already logged: matched-SGD CE/EM ratio, Adam stop-gradient arm, one larger SGD learning-rate rung, single-layer signature side-by-side.
4. **Use one canonical rule sentence.** Recommended: "EM conditioning survives when responsibility gradients reach the relevant parameters directly, when the EM path dominates the update, or through a verified compatible transport." If stochastic/simplex transport remains untested and only first-order, qualify it.
5. **Standardize terminology.** Use `corridor` for the learned dense CE-to-`W_1` route; use `path` only generically. Use `lambda` consistently instead of mixing `lambda` and `lambda_reg`, except where explicitly naming legacy experiment settings.
6. **Consolidate future work.** Keep future-strengthening material in `limitations.tex` (or a short future-work paragraph), not in setup and results.

## File-by-File Update List

### `supervised_study/draft/abstract.tex`

- **P0** Add the affine qualifier to the private-parameter curvature claim: "with private affine component parameters" or "with private affine energies."
- **P1** Replace "confirm the split named in the title" with non-self-referential, slightly softer wording such as "support a controlled dissociation: volume control transfers, conditioning does not."
- **P1** Replace "Adam becomes useful again" with more formal wording unless intentionally keeping it only in Results II. Example: "the advantage of adaptive methods reappears."
- **P1** Reconsider the final "simplex-compatible transport" condition. If kept, qualify it as a theoretical/first-order preservation route; otherwise drop it from the abstract.
- **P1** Keep the current "gradient diagnostics indicate" hedge unless matched-SGD ratios are added.

### `supervised_study/draft/introduction.tex`

- **P0** Make the affine assumption explicit at the first optimizer-conditioning mention, not only in Results in Brief.
- **P1** Replace "The experiments confirm the split" with softer wording such as "The experiments support the split" or "The experiments bear out the split in this controlled setting." This should be standardized with abstract and conclusion.
- **P1** Define `NLS` at first use as NegLogSoftmin.
- **P1** Soften the classical EM analogy: make clear that dense backpropagation is structurally parallel to a shared-parameter M-step, not literally an EM reduction for CE.
- **P1** Add one concrete preview of the operation taxonomy if the contribution list says the paper identifies preserving/destroying operations.
- **P1** Explain what "no detectable alignment" measures: cosine between CE and weighted local EM gradients at the intermediate distance interface, with a pointer to Results III.
- **P1** Motivate the `1/2`-saturation diagnostic before listing it as a contribution, or remove it from the contribution list.
- **P2** Consider merging the seven contributions into four broader contributions to reduce inflation.
- **P2** Replace "Log-sum-exp objectives are everywhere" with "Log-sum-exp objectives are ubiquitous" or similar.
- **P2** Add/verify citations for generalized EM roots, attention/contrastive/energy-based examples, and EM separability if not already covered elsewhere.

### `supervised_study/draft/background.tex`

- **P1** Soften the GMM log-determinant analogy. Use wording like "analogous in role to volume terms in mixture likelihoods" unless the precise direction of the log-det barrier is explained.
- **P1** Add a sentence that NLS makes the activations calibrated negative log-responsibilities, not just gradients. Note that this is why the no-NLS control is structurally different.
- **P1** Clarify whether diffuse responsibilities are repaired by volume control, diagnosed by `1/2` saturation, or both. Avoid implying variance/decorrelation directly fixes diffuseness unless supported.
- **P1** Make the "transport is not preservation" paragraph explicitly preview the formal operations/taxonomy in Theory.
- **P2** Add/verify citations for variance+decorrelation regularizers such as VICReg/Barlow Twins if those parallels are discussed.

### `supervised_study/draft/related_work.tex`

- **P0** If the text says Section Theory "makes precise" the shared-parameter M-step correspondence, soften to "makes structural" or "formalizes the parallel" unless a literal reduction is added.
- **P1** If basin/interpolation remains future work, shrink that subsection or explicitly frame it as motivation/future analysis.
- **P1** Align "formal sufficient condition" language with the actual theory: sufficient conditions for preserving local conditioning in private/unmixed settings, plus a guarantee-forfeiture result for dense corridors.
- **P1** Keep "drowned, not opposed" but avoid overclaiming from near-zero cosine; point to the calibrated limitation.
- **P2** Add/verify missing related-work references only if used: auxiliary losses/deep supervision, Git Re-Basin/permutation alignment, EM failure modes, mixtures of experts.

### `supervised_study/draft/notation.tex`

- **P1** The `1/2`-saturation diagnostic is already defined here. Add a cross-reference from Theory after Lemma 1 so readers do not miss the definition.
- **P1** Standardize `lambda` terminology. If `lambda_reg` remains for older ablations, explicitly say it is the same auxiliary weight convention.
- **P2** Add metric definitions here or in Experimental Setup, not only in captions: dead-unit threshold, redundancy formula, responsibility entropy construction.

### `supervised_study/draft/theory.tex`

- **P0** Resolve the `E||xtilde||^2` promise at lines 133--137. Either remove the promise or add the corresponding result/comparison in Results III.
- **P0** Address the full-auxiliary-objective gap. The clean curvature bound applies to the LSE term, but experiments train `L_LSE + L_var + L_tc`; explicitly scope the theory or add measured/bounded curvature for volume-control terms.
- **P1** In Proposition `prop:smooth-kernel`, explain that `phi' >= 0` is for first-order sign/component preservation, while the curvature bound uses bounded `|phi'|` and `|phi''|`; or drop monotonicity from the curvature proposition and keep it in the taxonomy.
- **P1** Add a local sentence after Lemma 1 defining the `1/2`-saturation diagnostic and equality geometry, even though notation already defines it.
- **P1** Move some interpretive prose out of the Proposition 2 proof into a post-proof paragraph or remark. Keep the honest scope, but make the formal statement/proof cleaner.
- **P1** Add a sentence in the classical EM connection that the supervised CE corridor is a structural parallel to shared-parameter EM, not a literal expected complete-data objective.
- **P2** Cite Böhning at the softmax/LSE Hessian bound and cite Popoviciu if keeping the named inequality.
- **P2** Replace "harmless convention" with a more formal phrase such as "up to the bias convention above."

### `supervised_study/draft/experimental_setup.tex`

- **P0** Remove or relocate `Future Strengthening Analyses`; consolidate this material in `limitations.tex`.
- **P0** If matched-SGD gradient ratios are not added, keep the Adam/SGD mismatch disclosure and ensure the abstract does not overstate the diagnostic.
- **P1** State why LayerNorm is used, since it affects CE/logit geometry and may interact with the variance runaway.
- **P1** State explicitly that the trained experimental model uses ReLU before distances, so it falls under the smooth/ReLU kernel caveat rather than the clean affine proposition.
- **P1** Add dead-unit threshold, redundancy definition, entropy definition, and final-epoch averaging protocol in the metrics subsection.
- **P1** State what the seeds control: initialization, data order, probe training, etc.
- **P1** Give the fixed-protocol linear probe details or point to an appendix/code location.
- **P2** Add/verify citations for linear probes and Hessian-vector products if the setup names them technically.

### `supervised_study/draft/results_volume_control.tex`

- **P1** Reframe the Var+Decorr-only comparison. The absence of NLS already proves no calibrated responsibility interface; use the numbers for structural differences, not to prove calibration by construction.
- **P1** Acknowledge metric-objective coupling: variance and redundancy metrics are close to the optimized losses. Then point to cross-metrics, especially Var-only worsening redundancy, as evidence the results are not trivial.
- **P1** Add standard deviations for Min Var and Accuracy in the capacity table if available. If unavailable, soften claims based on small accuracy differences.
- **P1** Explain the K=25 discrepancy between Table 1 and Table 2 as different seed counts/runs.
- **P1** Soften the "boundary refiners" explanation unless supported by evidence. Frame it as interpretation.
- **P2** If no K=64 lambda-recalibration run exists, soften the claim that fixed lambda is the reason for large-width accuracy tradeoff.
- **P2** Consider one sentence linking this section to prior single-layer numbers if calling it a replication.

### `supervised_study/draft/results_conditioning.tex`

- **P1** Add a side-by-side anchor for the prior single-layer optimizer signature in the same units, or make the prior-work citation explicit where "signature disappears" is claimed. This is a persuasiveness improvement rather than a correctness blocker.
- **P1** Add the plain baseline optimizer sweep if available; otherwise state that Experiment 3 tests the full EM-site model, not equivalence to a vanilla classifier.
- **P1** Address that the shown SGD sensitivity is low-learning-rate slowness, not high-learning-rate instability. Either add a higher learning-rate rung or phrase the result as convergence-speed/final-accuracy sensitivity.
- **P1** Explain the high-learning-rate Adam variance explosion and negative regularization loss. Mention possible LayerNorm scale-invariance only if supported; otherwise frame as an observed auxiliary escape direction.
- **P1** Define `Reg.` and negative values in the table caption. State which metrics are train-set and which are test-set.
- **P1** Add uncertainty for Min Var and Redundancy if available.
- **P2** Keep "Adam becomes useful again" here if desired, but remove/soften it elsewhere for register.

### `supervised_study/draft/results_mechanism.tex`

- **P0** Replace "Paper 2" with the actual prior-work citation or descriptive phrase.
- **P0** Resolve or explicitly acknowledge the `E||xtilde||^2` comparison here if the theory promise remains.
- **P0** If possible, add matched-SGD CE/EM ratios; otherwise keep the current diagnostic caveat and ensure headline claims remain hedged.
- **P1** Consider adding open/untested rows to the ledger, or replace checkmarks with "supported" so the ledger does not read like marketing.
- **P1** Add a cautious note that observed cosines are far smaller than the rough random-direction scale, but avoid interpreting this without a better null model.
- **P1** Address the self-normalization issue in "epoch to 95% of final LSE." Prefer absolute thresholds or trajectory plots; if not available, state the limitation.
- **P1** Draw out the `lambda=1` feature result: label gradients still flow, but EM dominance determines both conditioning and representation quality.
- **P1** Remove or relocate the `Future Strengthening Analyses` subsection to limitations.
- **P2** Replace bare checkmarks with text labels if the table looks too informal.

### `supervised_study/draft/discussion.tex`

- **P1** Use the canonical rule sentence and match abstract/conclusion wording.
- **P1** If the pretraining/fine-tuning subsection uses cosine as evidence for basin movement, remove that sentence or make clear local gradient direction is not basin evidence.
- **P1** Replace "predicted" vs "replicated" drift for the decoder-free volume-control result; use "replicated" where referring to prior observed ablations.
- **P1** If companion attention work is mentioned, cite it or mark it as in preparation.
- **P1** Consolidate future-work discussion with `limitations.tex`; avoid repeating the same TODOs in multiple sections.
- **P2** Promote the practical rule sentence if it is the cleanest practitioner takeaway.

### `supervised_study/draft/limitations.tex`

- **P0** Add a paragraph for the unresolved `E||xtilde||^2` comparison, unless it is removed from Theory or added to Results III.
- **P1** Add a metric-objective-coupling limitation for dead-unit/redundancy metrics in Results I.
- **P1** Add the one-sided learning-rate-sweep limitation: current sweeps mainly show low-learning-rate slowness, not an upper instability edge.
- **P1** Add an auxiliary-objective runaway limitation: high-learning-rate Adam can drive variance/regularization to extreme values; the clean curvature theory is about the LSE term unless full auxiliary curvature is measured.
- **P1** Add Adam-under-stop-gradient as an untested optimizer-signature completion unless that run is added.
- **P1** Convert draft-note voice ("the final presentation should...", "the paper should...") to paper voice ("we did not...", "we do not claim...", "a remaining check is...").
- **P2** In the cosine limitation, optionally add that observed values are smaller than the random-direction scale but are not further interpreted.

### `supervised_study/draft/conclusion.tex`

- **P0** Replace "confirm" with "demonstrate in this controlled setting" or "bear out."
- **P1** Add the affine qualifier to the private-parameter curvature sentence.
- **P1** Use the canonical rule sentence. Qualify or drop "simplex-compatible transport" unless the conclusion explicitly says it is a theoretical/untested first-order preservation class.
- **P1** Cite or mark the companion attention work as in preparation, or remove the dangling reference.
- **P2** Point "layer-wise direction" toward the local/decoupled learning references or phrase it as a design implication rather than a named direction.

### `supervised_study/draft/references.bib`

- **P0** Remove local warning comments and TODOs before submission, or move them to a private notes file.
- **P1** Verify existing bibliographic metadata, especially entries currently noted as filled from memory.
- **P1** Add only verified references needed by text updates. Likely candidates to verify: Popoviciu, VICReg/Barlow Twins, linear probes, Pearlmutter/HVPs, auxiliary classifiers/deep supervision, permutation alignment/Git Re-Basin, EM textbook/mixture degeneracy references, and Gauss-Newton/GGN references.

### `supervised_study/proofs/lemma1.py` and `lemma1.md`

- **P1** Rename the "Popoviciu step (symbolic)" output to "variance identity (symbolic)" unless the script actually verifies Popoviciu's inequality.
- **P2** Clarify in comments/output that the symbolic identity is a representative check, not a full symbolic proof for all `K`.

### `supervised_study/proofs/prop1.py` and `prop1.md`

- **P1** Remove the sign-agnostic `min(|H-Hn|, |-H-Hn|)` check. The Hessian sign is determinate: `H = -C(r) kron xt xt^T` under the paper's convention.
- **P2** Keep an explicit comment explaining the sign convention to avoid future regressions.

### `supervised_study/proofs/prop1_smooth.py` and `prop1_smooth.md`

- **P1** Replace the tautological ReLU absorbing-state check with an actual gradient-through-dead-unit numerical check, or remove the PASS line.
- **P2** Reject/retry random ReLU Hessian samples whose preactivations are within a small tolerance of the kink before finite differencing.

### `supervised_study/proofs/prop2.py` and `prop2.md`

- **P2** Remove the duplicate `r = softmax(-dv)` computation in `check_residual_is_real`.
- **P2** Consider reporting the residual/full norm ratio in the paper only if the random-instance scope is clearly stated. Do not imply the Gauss-Newton term is generally dominant or non-dominant from this diagnostic alone.

### `supervised_study/proofs/README.md`

- **P2** Add one sentence in the paper or reproducibility note that verification scripts exist. The README itself is fine after the script-output wording fixes.

## Update Order

1. Inventory available data/logs before rewriting around missing numbers: matched-SGD ratios, Adam stop-gradient, baseline optimizer sweep, higher learning-rate rung, single-layer side-by-side, capacity-table SDs, K=64 lambda recalibration, and probe variants.
2. Resolve true P0 consistency: `E||xtilde||^2`, abstract/introduction/conclusion overclaims, `Paper 2` leak, related-work "made precise" wording, and future-work consolidation.
3. Fix measurement-scope wording based on the data inventory: matched-optimizer caveat, one-sided LR sweep, metric-objective coupling, auxiliary runaway.
4. Clean theory/proof presentation: smooth-kernel monotonicity, Prop 2 prose placement, script sign/tautology issues.
5. Add available numbers/runs if they exist; otherwise soften claims or move items to limitations.
6. Verify and update citations last.
