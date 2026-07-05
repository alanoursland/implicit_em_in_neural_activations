# Proposal: Objective vs. Mechanism — The Bias Loophole and Why InfoMax Needs Competition

Source study: infomax_study/. Status: initial results exist (reports/initial_infomax_results.md); the designed sweep has not run.

## Thesis

InfoMax specifies *what* a representation should be (informative, independent features). It does not specify *how*, and gradient descent will satisfy the objective by the cheapest available path — which is not the intended geometry. The paper's central exhibit is the **bias loophole**: a ReLU layer achieves fully independent outputs with *redundant* weights, by giving identical hyperplanes different thresholds. Output independence improves while weight diversity worsens; pushing the independence penalty harder makes the loophole worse, not better. Competition (softmax / EM structure) closes the loophole structurally: units that share a direction produce near-identical pre-softmax values and are forced apart by the normalization. The claim: **InfoMax is an objective; implicit EM is the mechanism; the objective without the mechanism is satisfiable by geometry-free solutions.**

## Evidence in hand

- The loophole, demonstrated and characterized: redundancy drifts upward (0.29 → 0.54 over 100 epochs) while output correlation falls to 0.04; λ_tc = 100 drives redundancy to 7.7. Explicit weight regularization (λ_wr) patches it; softmax removes it with no explicit penalty at all.
- The counterproductive-pressure result: with softmax, high λ_tc *fights the simplex constraint* and degrades both metrics. With EM structure, the entropy term alone suffices.
- An early sighting of the conditioning phenomenon: without softmax, SGD needs lr ≈ 10 and Adam wins; with softmax, SGD ≈ Adam at lr = 0.001. This is the same effect later proved and measured in the failure-conditions work (paper_proposals/failure_conditions.md) — cite it as the mechanism rather than re-deriving.

## What is required

1. **Formalize the loophole.** It is provable: for a ReLU layer with K units sharing one weight direction and staggered biases, the outputs form a set of nested half-space indicators whose correlations can be driven arbitrarily low on any input distribution with sufficient spread along that direction. A short proposition with a constructive proof turns the exhibit into a theorem. Cost: days of writing; the construction is elementary.
2. **Run the designed sweep** (config/sweep.yaml exists: 6 activations × 3 widths × 10 seeds): does the pre-competition activation matter once softmax is present? Identity vs ReLU vs tanh. Cost: ~6 hours CPU per the study's own estimate.
3. **Scale and transfer checks.** K = 64/256; linear probes on the learned features. The loophole predicts probe accuracy should be *unrelated* to output independence in Architecture A and correlated in Architecture B — a falsifiable signature that the geometry, not the objective value, carries task information. Cost: ~1 day.
4. **Position against the collapse literature.** The loophole is a cousin of dimensional collapse in self-supervised learning (VICReg, Barlow Twins operate on outputs and are equally loophole-exposed in principle). One section, potentially one small experiment showing the loophole in a VICReg-style setup — high-upside if it holds, since it would show the finding is not an artifact of this framework. Cost: ~2 days, exploratory.

## Assessment

This is the smallest paper in the set — a focused, workshop-to-short-conference piece. Its risk is overlap: the "EM structure conditions optimization" half now belongs to the failure-conditions paper, and the "InfoMax is the volume control" half was published in Paper 2. What remains distinctly this paper's is the loophole itself and the objective/mechanism distinction, which neither published sibling states. Write it tight around that, cite outward for everything else.

## Out of scope

Volume control theory (Paper 2), conditioning theory (failure-conditions paper), multi-layer stacking (layerwise proposal).
