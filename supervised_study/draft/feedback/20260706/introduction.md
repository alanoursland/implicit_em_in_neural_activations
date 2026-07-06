This intro cashes the second promissory note from the abstract cleanly, and partially cashes the first. Let me go through it.

**The argument**

The "Main Claim" subsection is the best writing in the paper so far. The sharing-vs-dominance separation resolves the objection I raised at the abstract stage about bridging second-order theory to first-order measurements: you're explicitly saying the Hessian argument explains *why* the CE contribution is foreign (its curvature isn't LSE-controlled), and dominance explains *when* it matters. The sentence "This is why the claim is not that connectivity alone destroys EM conditioning" is exactly the pre-emptive move a good reviewer would otherwise force you to make. Keep that whole paragraph structure.

The parameter-independence question also mostly resolves: "with private affine component parameters, this yields a parameter-independent curvature bound" — with affine parametrization the Jacobian ∂d/∂θ is data-dependent but not parameter-dependent, so "bounded by a data constant" is right. Two residual points, though. First, the intro says "affine" in Results in Brief but the earlier subsection just says "private component parameters" — make the affine assumption explicit at first mention, because for non-affine distances (quadratic in μ, say) the claim is false and a careful reader will test it against the Mahalanobis case immediately. Second, the abstract still says "parameter-independent" without the affine qualifier; sync them.

The classical-EM analogy (shared parameters → coupled M-step) is a genuinely nice framing and I think it's correct as an analogy, but it's doing rhetorical work slightly beyond what's proven: in classical EM, parameter sharing couples the M-step but the update is still EM. Here the claim is stronger — the corridor contribution isn't EM *at all*. The analogy motivates, it doesn't establish. One sentence acknowledging that this is an analogy, not a reduction, would immunize it.

Remaining exposures a reviewer will find:

1. **"Simplex-compatible transport" from the abstract has vanished.** The intro says the identity extends "only when the path preserves component identity," which is vaguer. If contribution 2 ("identifies which operations preserve the invariant") delivers a characterization, the intro should preview it in one concrete clause — e.g., naming permutations/positive rescalings or whatever the actual class is. Right now the abstract promises a condition the intro doesn't restate.

2. **Alignment measurement.** "No detectable alignment" is carrying weight — it's what licenses the dilution story over an interference story. The intro repeats it twice without saying what was measured (cosine similarity of the two gradient fields? over training? at which layer?). One clause would help, and the body must be rigorous here because near-zero cosine in high dimensions is the *default* for unrelated vectors; a reviewer may say the null result is uninformative. If you have a calibrated baseline (e.g., cosine against random rotations of the EM gradient), preview it.

3. **Contribution 7 appears from nowhere.** The ½-saturation diagnostic hasn't been mentioned in the abstract or the intro body. Contributions lists shouldn't contain items the introduction never motivated. Either add a sentence in Results in Brief or cut it from the list and let the body introduce it.

4. **Seven contributions is too many.** Items 1–3 are the theory told three ways, 4–6 are the experiments told three ways. Reviewers read long contribution lists as inflation. I'd merge to four: (i) the local conditioning contrast + invariant-preserving operations [1+2], (ii) the at-site/corridor taxonomy [3], (iii) the transfer split with causal mechanism [4+5+6], (iv) the diagnostic [7, if kept].

**Style**

Much less to do here than you might expect — this is already academic register. Specific spots:

- *"Log-sum-exp objectives are everywhere in neural networks."* — "everywhere" is casual for an opening sentence. "Log-sum-exp objectives pervade neural networks" or "...are ubiquitous in..." Fine to keep some punch; just not "everywhere."
- *"Those results invite a tempting inference"* — good sentence, keep. This is the un-AI kind of directness.
- *"Adam is again useful"* — same note as the abstract; it recurs. If you want to keep the parallelism ("SGD is again learning-rate sensitive and Adam is again useful"), it's borderline acceptable as a deliberate stylistic signature, but pick one register and hold it — the abstract and intro currently both use it, and a third occurrence in the results section will start reading as a tic.
- *"the split named in the title"* — doesn't appear here, good; make sure the abstract fix happens.
- "NLS transform" appears undefined. Expand at first use.
- The em-dash count is low and the paragraphs have varied rhythm — this doesn't read AI-generated to me. The one tell-adjacent habit is triadic lists ("die, collapse, or duplicate"; "local, graded, and structural"; "directly, dominantly, or through..."). The load-bearing triads earn their place; just don't add more.

**References**

The intro is citation-thin in three places:

- **Opening paragraph** ("softmax classification, attention, contrastive losses, energy-based objectives"): attention wants Vaswani et al. 2017; contrastive wants InfoNCE — Oord et al. 2018, *Representation Learning with Contrastive Predictive Coding* (high confidence); energy-based wants LeCun et al. 2006 energy-based learning tutorial (high confidence it exists, moderate on exact title). One bracket of citations after that sentence.
- **"implicit generalized EM" paragraph**: needs Neal & Hinton 1998 (*A View of the EM Algorithm that Justifies Incremental, Sparse, and Other Variants* — high confidence) next to your self-citations, both to anchor "generalized EM" and to show the framing has classical roots. Xu & Jordan (1996, *On convergence properties of the EM algorithm for Gaussian mixtures* — moderate confidence on title) is relevant to the EM-as-conditioned-gradient connection.
- **Classical EM fact about separable M-steps**: this is textbook material but stated as a load-bearing premise, so cite it — Dempster, Laird & Rubin 1977 for EM itself, and McLachlan & Krishnan's EM book (*The EM Algorithm and Extensions*, moderate confidence) for the separability discussion.
- The Gauss–Newton/conjugated-Hessian decomposition has precedent in Martens' work on second-order optimization (Martens 2010/2014, and Martens & Grosse K-FAC 2015 — high confidence these exist, moderate on which is the right fit). Worth citing when the composed Hessian appears, to show you know the GN term is standard and your contribution is the contrast, not the decomposition.

Net assessment: the intro is structurally sound and the sharing/dominance distinction is the paper's best idea, clearly stated. The exposures are all fixable at the sentence level except #2 (alignment), which is a body-of-paper obligation — that's the one place the argument could actually be attacked rather than just nitpicked. Send the theory section next if you want; that's where I'd check whether the invariant-preservation claim (contribution 2) gets a real characterization or just examples.