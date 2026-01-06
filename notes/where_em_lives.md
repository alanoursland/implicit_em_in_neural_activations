# Where EM Lives

## The Claim and Its Scope

The implicit EM result establishes an algebraic identity: for objectives of the form L = log Σⱼ exp(−dⱼ), the gradient with respect to each distance is exactly the negative responsibility:

```
∂L/∂dⱼ = −rⱼ
```

This is exact. No approximations. It holds whenever the loss has log-sum-exp structure over distances and the distances are differentiable.

But where does this identity apply? What part of a neural network is "doing EM"?

## The Output Interface

The identity holds at the interface between model outputs and loss function. The model produces distances (or logits, which are negative distances). The loss operates on those distances. The gradient of the loss with respect to each distance is the responsibility.

For cross-entropy classification:
- Model outputs logits z₁, ..., zₖ
- Loss is L = −z_y + log Σₖ exp(zₖ)
- Gradient: ∂L/∂zⱼ = softmax(z)ⱼ − 𝟙[j = y] = rⱼ − 𝟙[j = y]

The responsibilities rⱼ = softmax(z)ⱼ appear in the gradient. The output layer receives responsibility-weighted updates. This is where EM lives.

## What Backpropagation Delivers

Now trace the gradient backward. The first hidden layer receives:

```
∂L/∂W₁ = ∂L/∂z · ∂z/∂a₁ · ∂a₁/∂z₁ · ∂z₁/∂W₁
```

The term ∂L/∂z contains the responsibilities. They propagate backward through the chain rule. In this sense, every parameter update is "responsibility-weighted"—the responsibilities from the output scale the gradients throughout.

But this does not mean the first layer is doing EM.

## What the First Layer Is Not Doing

The first layer computes z₁ = W₁x + b₁. Each row of W₁ defines a hyperplane in input space. These are not prototypes in the EM sense. They do not compete for responsibility over inputs. There is no E-step that assigns inputs softly to hyperplanes.

The gradient to W₁ tells each hyperplane how to move to reduce the final loss. If hyperplane 3 and hyperplane 7 both help, both receive gradient. They may converge to the same orientation. Nothing in the first layer's structure prevents this.

The responsibilities exist at the output. They scale the gradient signal. But by the time that signal reaches the first layer, it has been transformed through many intermediate Jacobians. The first layer sees "move this direction to reduce loss," not "you have responsibility 0.3 for this input."

## The Asymmetry

There is a fundamental asymmetry between the output layer and internal layers:

**Output layer:**
- Computes distances to class prototypes (logits)
- Softmax/LSE creates competition among classes
- Gradients are responsibilities
- Prototypes receive responsibility-weighted updates
- EM interpretation applies directly

**Internal layers:**
- Compute features via affine transformation + nonlinearity
- No competition among units (in ReLU networks)
- Gradients are backpropagated chain rule terms
- No prototype interpretation
- EM interpretation does not apply

The output layer is a mixture model. The internal layers are a learned feature map. The feature map is trained by backprop through the mixture model's loss, but it is not itself a mixture model.

## A Precise Statement

The implicit EM framework applies to the following:

1. A set of distances or energies d₁, ..., dₖ computed for an input
2. A log-sum-exp objective over those distances
3. Gradient-based optimization

Under these conditions, gradient descent performs EM: responsibilities emerge as gradients, and parameters producing distances receive responsibility-weighted updates.

In a typical neural network:

- The final linear layer produces distances (logits)
- Cross-entropy is the LSE objective
- The final layer's weights are prototypes receiving responsibility-weighted updates

The layers before the final layer are outside this framework. They produce the representation on which the mixture model operates, but they do not themselves have mixture structure.

## Why This Matters

Misunderstanding where EM lives leads to incorrect claims:

**Incorrect:** "The whole network is doing EM."
The whole network is doing gradient descent. Only the output layer has EM structure.

**Incorrect:** "Internal representations are mixture components."
Internal representations are features. They may be useful for the mixture model at the output, but they are not themselves mixture components with responsibilities.

**Incorrect:** "EM explains why internal layers specialize."
EM explains why output classes specialize. Internal layers specialize (when they do) for other reasons—or fail to specialize, leading to redundancy.

**Correct:** "The loss geometry imposes EM structure at the output, and gradients from this structure flow backward to shape internal representations."

## The Implication

If we want EM structure throughout the network—competition within layers, responsibility-weighted updates to internal features, specialization by construction—we cannot get it from the output loss alone.

The output loss creates competition among outputs. That competition does not propagate backward through ReLU. Internal layers receive gradient but not competition.

To get competition within layers, something must change:
- Auxiliary losses at each layer (explicit layer-wise EM)
- Activation functions with competitive gradient structure (implicit layer-wise EM)
- Architectural changes that introduce normalization within layers

The observation that internal layers can be redundant while output layers specialize is not a failure of training. It is a consequence of where EM lives and where it does not.