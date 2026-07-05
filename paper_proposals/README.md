# Paper Proposals

Scope documents for papers in the implicit-EM program. Published so far:

- **Paper 1** — the gradient–responsibility identity (companion theory paper).
- **Paper 2** — *Deriving Decoder-Free Sparse Autoencoders from First Principles*, arXiv:2601.06478.

## Proposals, in recommended order of execution

**`failure_conditions.md`** — The conditions under which the EM property fails: local, graded, structural. The supervised study's paper (absorbs experiments 1–4). Closest to complete: theory and headline experiments exist; needs the Adam arm, the λ-resolution curve, formal tightening, and the basin analysis.

**`encoder_followups.md`** — Triage of the published paper's nine future directions into two standalone papers (LSE as a free anomaly detector; explicit EM with closed-form M-steps), plus threads absorbed by other proposals.

**`infomax_activations.md`** — The bias loophole: InfoMax is an objective, EM structure is the mechanism, and the objective without the mechanism is satisfied by geometry-free solutions. Smallest paper; initial results exist; needs the designed sweep and a short constructive proof.

**`layerwise_em.md`** — Deep networks as stacks of protected local EM sites; the ½-saturation diagnostic per layer answers "how far in does the EM property hold" at depth. Gated on the failure-conditions paper (uses its diagnostic and locality result).

**`attention_em.md`** — Attention sinks as mixture collapse; volume control for keys and heads. Predicts the at-site/structural split transfers to a third architecture. Experiments live partly in the related sink-suppression project; the data-dependent-components extension is the theory content.

## Dependency sketch

```
Paper 2 (published)
   └─ failure_conditions ── layerwise_em
   │        └───────────────── attention_em (taxonomy + stochastic-map arm)
   ├─ encoder_followups (A: anomaly, B: explicit EM)   — independent
   └─ infomax_activations                              — independent, smallest
```
