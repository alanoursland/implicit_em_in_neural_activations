The conclusion is the right length and does its job — restates the boundary without re-arguing it, and the two forward-pointing consequences give it somewhere to land. A few notes, then the promised full-paper consistency pass.

**The conclusion itself**

- *"The supervised experiments confirm the split from both sides"* — same note as the abstract: "confirm" from MNIST-only evidence is stronger than the limitations section (correctly) licenses. "Bear out" or "demonstrate in a controlled setting." The conclusion and abstract are where reviewers check whether your claims match your caveats; both currently claim slightly harder than the limitations concede.
- *"is restored, causally, by cutting the corridor or letting the EM path dominate"* — good; this is the paper's finding in one clause and it's stated accurately.
- The transferable-rule sentence lists "directly, through a simplex-compatible transport, or with enough dominance" — this is now the third distinct ordering/wording of the three conditions (abstract: "directly, dominantly, or through simplex-compatible transport"). Pick one canonical phrasing and use it verbatim in abstract, discussion rule, and conclusion. When the paper's takeaway sentence appears three times with variations, readers wonder if the variations are meaningful.
- "Simplex-compatible transport" is asserted as a survival route, but the paper's own limitations section says the stochastic-map arm is unrun and not load-bearing. Strictly, the theory shows first-order preservation only (mass and nonnegativity, not curvature — your own "partial preservation class" labeling). The conclusion is the worst place to overclaim a taxonomy cell you explicitly declined to test. Either qualify ("or, plausibly, through simplex-compatible transport — the untested preservation class") or drop it from the rule here and let the taxonomy section carry it.
- *"the layer-wise direction"* and *"the companion attention work"* — both dangling references again. The companion work needs a citation or an "in preparation"; "the layer-wise direction" should point at the local-learning citations you already have (Hinton 2022, Belilovsky 2019).
- Last sentence is good — "which properties transfer, which fail, and which interventions make them survive" is a clean closing triad and accurately inventories the paper.

**Full-paper consistency pass**

Promises made vs. cashed:

1. **E‖x̃‖² / descent-lemma comparison** — promised in Theory ("the experiments therefore report..."), never delivered, never acknowledged. *The only outright broken promise in the paper.* Decide: run, or delete the promise.
2. **"Made precise in Section 3"** (Related Work, shared-parameter M-step) — partially cashed; the Q(θ,ψ) formalization is structural parallel, not reduction. Soften "precise" or add the one-sentence concession in Theory.
3. **"Formal sufficient condition"** (Scope) vs. "identifies which operations" (intro) vs. the preserving/destroying taxonomy (Theory) — delivered at the taxonomy level; align the three phrasings to the weakest accurate one, and fix the unused monotonicity hypothesis in Prop 1′.
4. **½-saturation diagnostic** — used in Theory's Table 1 and contribution 7, defined only implicitly in Results III. Needs its two-sentence definition after Lemma 1.
5. **1/√K cosine baseline** — promised in Setup ("should compare"), delivered in Results III and Limitations. Change Setup's "should" to "is."
6. **Matched-SGD ratio, Adam-stop-gradient, denser λ curve, basin analysis, stochastic map** — each disclosed 2–3 times (Setup, Results III, Limitations, Discussion). Consolidate to Limitations only; delete both "Future Strengthening" blocks.

Terminology drift:

- **corridor/path** — both used throughout for the same object; standardize (corridor as the term, path informally at most).
- **λ / λ_reg / λ=0.001** — three notations; unify.
- **NLS / NegLogSoftmin** — define once at first use (intro currently uses bare undefined "NLS").
- **The three-condition rule** — three wordings (above).
- **"replication" vs. "prediction"** for the decoder-free ablation correspondence — Results I says replication, Discussion says predicted; use replication.
- **"Paper 2"** in Results III — internal name; replace with the citation.

Abstract re-audit against the body:

- "parameter-independent curvature bound" → add "affine" (Prop 1's actual hypothesis; the experiments themselves live under the kernel case, which the setup should also state).
- "Experiments... confirm the split named in the title" → restate the split, drop the self-reference, soften "confirm."
- "30–70×" → currently rests on the mismatched-optimizer diagnostic; either the matched run backs it or the abstract needs the hedge the body carries.
- "no detectable alignment or opposition" → accurate as bounded by Limitations; fine.
- "Adam becomes useful again" → keep at most once (Results II is where it earns it).
- "simplex-compatible transport" → same qualification issue as the conclusion.

Numbers cross-check: Table 1 vs Table 2 at K=25 (20.5 vs 19.0 redundancy etc. — caption note re: seed counts); Results III quotes spreads 10.9/5.9/1.3/1.2 consistently with its own table; ledger's "13–100×" vs prose "roughly 13–100×" consistent; abstract's "30–70×" matches Table's 30–71 ranges, fine.

And the run-list stands as before: matched-SGD ratio, Adam-stop-gradient arm, one higher η rung, K=64 λ-recalibration (or soften the two-regimes causal story), kNN/MLP probe check, E‖x̃‖² decision. All cheap relative to what's already done; none structural.

That's the complete pass. The paper's core is sound — the λ=1 arm and the curvature dichotomy carry it, the theory is verified three ways, and the limitations section is genuinely strong. What remains is bookkeeping and a handful of cheap runs. If you revise and want another pass over the diff, or want help drafting any of the specific fixes (the ½-saturation definition, the consolidated limitations paragraphs, the canonical rule sentence), send them along.