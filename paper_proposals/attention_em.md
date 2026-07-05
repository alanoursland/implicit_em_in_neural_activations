# Proposal: Attention Sinks as Mixture Collapse — Volume Control for Attention

Source material: notes/relationship_to_attention.md, notes/open_world_and_rejection.md, the failure-conditions taxonomy (paper_proposals/failure_conditions.md), and the published volume-control results (arXiv:2601.06478). Experiments for this paper live partly in a related project (the K = W_K·X sink-suppression work); this document states the theory frame and what a complete paper needs regardless of which repo runs it.

## Thesis

Attention is the one standard architecture component that is *explicitly* implicit EM: softmax over key similarities is the E-step, the value-weighted sum is the posterior mean. It therefore inherits the framework's failure catalog — and **attention sinks are mixture collapse wearing transformer clothes**. Softmax is a closed-world normalization: the responsibility mass must go somewhere, and nothing in the training objective prevents one key from becoming the component that claims everything at zero cost. That is precisely the degeneracy the log-determinant prevents in GMMs and InfoMax prevents in the decoder-free encoder. The paper's claim: **the volume control that stabilizes mixture components stabilizes attention** — variance penalties against sink/dead keys, decorrelation against redundant heads — and the framework predicts *which* parts transfer and which do not.

## The two predictions (from the at-site / structural taxonomy)

1. **Sink suppression should work.** Collapse is an *at-site* failure, and InfoMax penalties applied at the attention logits/keys act at the site of the softmax. The supervised study showed at-site volume control transfers even through a hostile corridor (dead units eliminated at every width and learning rate). Prediction: variance/decorrelation on attention scores or keys suppresses sinks without needing any architectural change.
2. **The optimization conditioning should not come along.** Keys are not private parameters — they are produced by one shared map W_K applied to data-dependent tokens, and the conditioning property was proved to require parameter privacy. Prediction: sink suppression succeeds while training dynamics stay ordinary (Adam still needed, learning-rate sensitivity unchanged). If the related project observes exactly this split, that is not a partial failure — it is the taxonomy confirmed in a third architecture.

## What is genuinely new territory (the research content)

- **Data-dependent components.** Every result in this repo assumes fixed component parameters. Attention's hypotheses (keys) change per input. The lemma does not cover this; extending the at-site claim to data-dependent component sets is the paper's theory contribution. The natural route: condition on the input, treat the per-input key set as a frozen mixture, and ask what the penalties control in expectation over inputs.
- **Multi-head structure as grouped competition.** Heads are parallel EM problems with shared token inputs — the grouped-softmax case from notes/candidate_competitive_activations.md. Head redundancy (well documented in the literature) is the cross-head analog of component redundancy; decorrelation *across heads* is the off-diagonal volume term at a new granularity.
- **The simplex-transport observation.** The attention matrix is row-stochastic; its transpose preserves gradient mass and non-negativity under pullback. Attention's value path is the one standard layer that transports responsibility structure rather than destroying it. If the stochastic-map arm of the failure-conditions work shows partial conditioning survival, this paper inherits a striking claim: transformers work partly because attention is the only simplex-compatible corridor in deep learning. Keep it as a conjecture section unless that experiment lands.

## What is required

1. **Sink phenomenology under the mixture lens** (~1 week): take small pretrained transformers (GPT-2 scale or smaller, or train 2–4 layer models), measure sink mass, key-space geometry, and responsibility entropy per head; show sinks look like collapsed components (one key at near-zero "distance" to everything).
2. **The intervention** (~2 weeks, likely in the related project): variance + decorrelation penalties on scores/keys during training (or fine-tuning); measure sink mass, head redundancy, perplexity cost. The λ dial from the failure-conditions work applies directly — expect a trade-off curve, report it honestly.
3. **The split test** (~1 week): alongside the intervention, measure conditioning (lr sensitivity, Adam vs SGD gap) to test prediction 2. This is what makes the paper a theory test rather than a regularizer ad.
4. **Theory section**: the data-dependent extension above; the closed-world statement for attention (a sink is the system's only way to express "no key matches" — connecting sinks to the rejection problem is likely their *function*, which is why suppression must be paired with an escape hatch such as a learned null key — this may be the paper's most interesting discussion).
5. **Positioning**: attention-sink literature (StreamingLLM etc.), head-redundancy/pruning literature, attention-as-EM prior work cited in arXiv:2601.06478.

## Risks

- Sinks may be functional (the null-hypothesis slot), and naive suppression may hurt models — the escape-hatch variant (null key + volume control on the rest) is the hedge, and finding that suppression *without* an escape hatch fails would itself confirm the closed-world analysis.
- Transformer-scale experiments are a different cost regime from everything else in this program; the small-model phenomenology (item 1) must carry the paper if compute is limited.
