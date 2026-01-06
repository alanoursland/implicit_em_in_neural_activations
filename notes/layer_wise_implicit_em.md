# Layer-Wise Implicit EM

## The Current Situation

The implicit EM result establishes that for objectives with log-sum-exp structure, ∂L/∂dⱼ = −rⱼ. Gradient descent performs EM at the interface between model outputs and loss function. The output layer's prototypes receive responsibility-weighted updates.

But what about internal layers?

When gradients backpropagate through a deep network, the first layer receives:

∂L/∂W₁ = Σⱼ rⱼ (∂dⱼ/∂W₁)

This is responsibility-weighted in the sense that the output responsibilities scale the gradients. But W₁ is not a set of prototypes. The first layer computes features that, after many transformations, become distances at the output. The EM interpretation does not attach to internal structure.

The internal layers are trained by backprop through an EM-structured loss. They are not themselves performing mixture inference. There is no E-step at layer 3. There are no responsibilities over layer 3's units.

## The Consequence: No Internal Competition

Consider a ReLU network. Each hidden unit defines a hyperplane. The gradient to each hyperplane depends on how it affects the final loss, but hyperplanes do not compete with each other for responsibility over inputs.

Nothing prevents multiple hyperplanes from learning the same cut. If hyperplane 3 and hyperplane 7 both contribute to reducing the loss, both receive gradient. They may converge to nearly identical solutions. This is observed empirically: trained networks exhibit massive redundancy, with many units learning overlapping or identical features.

At the output layer, softmax/LSE structure creates competition. If class A takes responsibility, class B loses it. Output prototypes must specialize. But this competition does not propagate backward through ReLU. The internal layers inherit no such pressure.

## The Question

Can we induce implicit EM at each layer, not just at the output?

If so, each layer would have its own responsibility structure. Units within a layer would compete for inputs. Specialization would be forced throughout the network, not just at the head.

## Two Approaches

### Approach 1: Greedy Layer-Wise Pretraining

Train each layer independently with its own LSE objective before stacking.

Layer 1 sees raw inputs. It has K₁ prototypes. Train with L₁ = −log Σⱼ exp(−d₁ⱼ) until convergence. The prototypes specialize to cover modes in the input distribution.

Freeze layer 1. Layer 2 sees layer 1's outputs. Train with L₂ = −log Σⱼ exp(−d₂ⱼ). The prototypes specialize to cover modes in the transformed space.

Continue stacking. Optionally fine-tune end-to-end afterward.

This approach gives per-layer EM guarantees during pretraining. Each layer's prototypes are forced to differentiate. But:

- Local objectives may not align with global task. A layer might specialize beautifully for structure irrelevant to the output.
- Greedy training discards information about downstream utility.
- Fine-tuning may undo the specialization if the end-to-end loss doesn't preserve it.

This echoes deep belief net pretraining, which worked but became unnecessary with better optimization and architectures.

### Approach 2: Activation Structures with Competitive Gradients

Design activations such that the backward pass through those activations has responsibility-weighted structure. No auxiliary losses. The competition is embedded in the architecture.

If a layer computes z = Wx + b and applies an activation a = f(z), the gradient to z is:

∂L/∂z = (∂L/∂a) ⊙ f'(z)    [for element-wise activations]

or

∂L/∂zᵢ = Σⱼ (∂L/∂aⱼ)(∂aⱼ/∂zᵢ)    [for activations with cross-unit dependencies]

ReLU is element-wise. The Jacobian is diagonal: ∂aᵢ/∂zⱼ = 0 for i ≠ j. No cross-unit interaction. No competition.

Softmax has cross-unit terms: ∂aᵢ/∂zⱼ = aᵢ(δᵢⱼ − aⱼ). Raising zⱼ raises aⱼ and lowers all other aᵢ. Competition exists.

The question: can we design an activation that has competitive gradient structure while preserving the properties needed for deep learning (expressiveness, stable gradients, magnitude information)?

## The Tradeoff

Competition buys specialization but may cost expressiveness.

ReLU allows combinatorial activation patterns. With K units, up to 2^K regions can be distinguished (in principle). Many units active simultaneously. Rich representational capacity.

Softmax forces soft selection. The output lies on the simplex. Effectively one unit "wins" (softly). Limited to K regions with interpolation.

An activation that introduces competition while preserving expressiveness would be valuable. Candidates exist but are not yet validated:

- z ⊙ softmax(z): element-wise product preserves magnitude while softmax provides competition
- log softmax(z) = z − logsumexp(z): subtracts shared value, retains relative magnitudes
- Grouped competition: softmax within groups, independence across groups

## Open Questions

1. **Does competitive activation actually reduce hyperplane redundancy?** Empirical question. Testable with single-layer experiments on 2D data where we can visualize the learned hyperplanes.

2. **What is the right output for a competitive layer?** If each layer does soft assignment, what does it pass forward? Responsibilities lose magnitude. Raw activations after competition may work. Needs investigation.

3. **How does competition interact with depth?** If layer 1 forces specialization, does layer 2 see a "better" input distribution? Or does the simplex constraint impoverish the representation?

4. **Volume control.** Softmax without regularization allows collapse—one unit dominates by growing weights unboundedly. GMMs prevent this with log-determinant terms. What is the analogous mechanism for competitive activations? Weight normalization? Learned temperature? Explicit volume penalty?

5. **Does end-to-end training preserve layer-wise competition?** If we use competitive activations and train end-to-end with a standard loss, do the internal layers maintain specialization? Or does the global objective override local competition?

## Relationship to Attention

Attention is a competitive mechanism. Softmax over scores creates responsibilities. Values are combined by responsibility-weighted sum. The backward pass through attention has the structure we want.

But attention operates over input-derived keys and values, not learned prototypes. And it's expensive—O(n²) in sequence length.

The question of "implicit EM activations" might be asking: can we get attention-like competition among learned prototypes, cheaply, as a drop-in activation replacement?

This may or may not be the right framing. Attention preserves information by outputting a weighted combination of high-dimensional values. A scalar activation necessarily discards information. The comparison may be inapt.