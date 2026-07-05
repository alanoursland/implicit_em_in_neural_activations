# Proposal: Layer-Wise Implicit EM — Deep Networks as Stacks of Local Mixture Sites

Source material: notes/layer_wise_implicit_em.md, notes/where_em_lives.md, the locality result (paper_proposals/failure_conditions.md), arXiv:2601.06478. This is the natural successor to the failure-conditions paper and the second (and likely last) paper the supervised study line supports.

## Thesis

The failure-conditions work established that the EM property is local: it exists at exponentiation+normalization sites and extends zero layers through learned dense maps. The constructive corollary has not been tested: **a deep network can be built as a stack of local EM sites** — each layer a mixture model over the previous layer's output, each with its own LSE + volume control, trained with stop-gradients between sites (each layer sees only its own objective) or jointly with dominance-balanced λ. The question: do local sites stay healthy under depth, and does the stack learn a usable hierarchical mixture — hierarchical prototypes, all the way up?

This is greedy layer-wise pretraining, rehabilitated with the objective it always lacked. Reconstruction was an awkward layer-wise objective; LSE + InfoMax is a principled one. And the mechanism story now explains *why* end-to-end fine-tuning erased pretraining's benefits historically (dominant orthogonal gradient overwrites the pretrained structure — measured in the failure-conditions work and in the basin experiments), which predicts the design rule this paper tests: **protect each site or lose it.**

## The instrument

The ½-saturation diagnostic from the failure-conditions work: ‖∇²_d L_LSE‖ at a site measures mixture sharpness against an absolute scale (½ = fully formed two-way competition; the bound is exact and parameter-free). Plotting the diagnostic per layer vs depth is the paper's central figure: it answers "how far into a model does the EM property hold" *quantitatively* — per site, under joint vs stop-gradient training, as depth grows.

## Design (core experiment)

Depth ∈ {2, 3, 4, 6} stacked EM layers (Linear → nonneg kernel → LSE+var+tc per layer), three training regimes:
1. **Greedy**: train layer ℓ to convergence on layer ℓ−1's frozen output.
2. **Joint-protected**: all layers simultaneously, stop-gradient between sites (each layer's parameters receive only its own site's objective).
3. **Joint-unprotected**: all layers simultaneously, gradients flow (each layer sees its own site + everything above).

Per layer, per regime: the ½-diagnostic, dead/redundant components, responsibility entropy; global: linear probe per depth, conditioning sweep (does lr-insensitivity hold per site in regimes 1–2 and fail in 3?). Then one supervised variant: classifier head on top, testing whether a protected hierarchical mixture supports classification competitively with an end-to-end MLP of matched size — expect a gap (the 88-vs-96 trade-off will recur); measure whether it *shrinks with depth*, which is the interesting question: does hierarchy buy back task alignment that single-layer mixtures lack?

## What is required

1. The stack implementation and the three regimes — mostly assembled from existing supervised_study code (~3 days).
2. The depth × regime sweep on MNIST + Fashion-MNIST (~CPU days; all small models).
3. A theory question worth settling first: what does layer ℓ's mixture see? Layer ℓ−1's output is a distance/responsibility representation — non-negative, roughly calibrated after NLS. Whether LSE over *distances-of-distances* has sensible semantics (mixture over mixture assignments — approaching a hierarchical mixture model / deep GMM) deserves a short note before the code is written; the deep-GMM literature (van den Oord & Schrauwen) is the anchor.
4. Positioning: DBN/greedy pretraining history, deep GMMs, target propagation and local-learning literature (Hinton's forward-forward is the obvious contemporary comparison — local objectives, no backprop between blocks; the EM framing supplies what forward-forward lacks: a derived per-layer objective).

## Assessment

Should not start until the failure-conditions paper is drafted: it consumes that paper's diagnostic, its locality claim, and its trade-off framing. If the depth sweep (item 5 of the failure-conditions required list) produces a clean per-layer figure, that figure can seed this paper directly. Risks: distances-of-distances semantics may be genuinely muddy (item 3 is the gate); the probe-accuracy gap may not shrink with depth, in which case the paper is a negative result about hierarchical mixtures — publishable within this program's falsificationist pattern, but plan for it.

## Out of scope

Attention/transformers (attention_em.md), convolutional variants, anything at scale beyond small image datasets.
