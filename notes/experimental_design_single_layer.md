# Experimental Design: Single Layer

## Objective

Test whether competitive activations reduce hyperplane redundancy in a single layer. This is the simplest setting that can reveal whether competition forces specialization.

## Setup

### Data

2D input data. Low dimension allows visualization of learned hyperplanes.

Several data configurations:

1. **Gaussian clusters.** 4-8 well-separated clusters. Clear mixture structure. Ideal case for prototype learning.

2. **Concentric rings.** Not linearly separable. Tests whether competition helps or hurts on non-mixture-like structure.

3. **Uniform square.** No structure. Baseline for how activations partition space with no signal.

4. **Polytope targets.** Data labeled by whether it falls inside a convex polytope. Matches the setting from the hyperplane visualization. Ground truth requires specific hyperplanes.

### Architecture

Single hidden layer autoencoder:

```
x ∈ ℝ² → z = Wx + b ∈ ℝᴷ → a = f(z) ∈ ℝᴷ → x̂ = W'a + b' ∈ ℝ²
```

- Input dimension: 2
- Hidden dimension K: vary from 4 to 32
- Activation f: the experimental variable
- Output: reconstruction of input

Autoencoder rather than classifier because:
- No labels required (unsupervised)
- Forces layer to capture structure useful for reconstruction
- Reconstruction error is interpretable
- No confound from output layer competition

### Activations to Test

1. **ReLU** — Baseline. No competition. Expect redundancy.

2. **Softmax** — Maximum competition. Expect specialization. Expect poor reconstruction (magnitude loss).

3. **Log-softmax** — Moderate competition. Relative magnitudes preserved. Middle ground.

4. **z ⊙ softmax(z)** — Competition with magnitude. Novel candidate.

5. **Grouped softmax (groups of 2)** — Pairwise competition. Intermediate expressiveness.

6. **Grouped softmax (groups of 4)** — Stronger local competition.

### Volume Controls to Test

For each competitive activation, test with and without:

1. **None** — Baseline. See if collapse occurs.

2. **Weight normalization** — Rows of W normalized to unit norm.

3. **Weight decay** — Standard L2 penalty on W.

4. **Entropy regularization** — Penalty for low-entropy responsibility distributions: -λ H(softmax(z)).

5. **Orthogonality regularization** — Penalty ||WᵀW - I||².

### Training

- Optimizer: Adam, learning rate 1e-3
- Batch size: 64
- Epochs: 500 (enough to converge)
- Loss: MSE reconstruction + any regularization terms
- Multiple seeds: 10 runs per configuration for variance estimates

## Metrics

### Hyperplane Redundancy

Each row of W defines a hyperplane in 2D (a line). Measure redundancy:

**Angular similarity:** For each pair of rows wᵢ, wⱼ, compute:
```
cos(θᵢⱼ) = wᵢ · wⱼ / (||wᵢ|| ||wⱼ||)
```

High |cos(θᵢⱼ)| means similar or opposite directions—redundant.

**Redundancy score:** Average pairwise |cos(θᵢⱼ)| over all i ≠ j. Lower is better.

**Effective rank:** Eigenvalues of WᵀW. How many significant directions?
```
eff_rank = (Σᵢ σᵢ)² / Σᵢ σᵢ²
```
Ranges from 1 (collapsed) to K (full rank). Higher is better.

**Cluster count:** Cluster the weight vectors by direction. How many distinct clusters? Fewer than K indicates redundancy.

### Reconstruction Quality

**MSE:** Mean squared error on held-out data. Lower is better.

**Per-cluster MSE:** If data has clusters, measure reconstruction error per cluster. Does the model capture all clusters or sacrifice some?

### Specialization

**Responsibility entropy per input:** For each input x, compute responsibilities r = softmax(Wx + b). Compute entropy H(r). Average over dataset.
- Low entropy: sharp assignment, one unit dominates per input
- High entropy: diffuse assignment, many units active

**Responsibility entropy per unit:** For each unit j, compute average responsibility across dataset. Compute entropy of this distribution.
- Low entropy: some units always win, others never win (collapse)
- High entropy: responsibilities distributed across units (healthy)

**Input-unit assignment map:** For each input, record which unit takes max responsibility. Visualize as colored scatter plot. Are regions cleanly partitioned?

### Collapse Detection

**Maximum average responsibility:** For each unit j, compute E[rⱼ] across dataset. Report max. If close to 1, one unit dominates—collapse.

**Dead units:** Count units with E[rⱼ] < 0.01. High count indicates collapse has starved units.

**Weight norm variance:** Variance of ||wⱼ|| across units. High variance with weight normalization off may indicate incipient collapse.

### Training Dynamics

**Loss curves:** Track reconstruction loss over training. Competitive activations may have different convergence properties.

**Responsibility evolution:** Track average responsibilities per unit over training. Do units differentiate early or late? Does one unit start dominating?

**Weight angle evolution:** Track pairwise angles between weight vectors over training. Do they spread out or cluster?

## Visualization

For 2D data, visualization is the primary tool for understanding.

### Hyperplane Visualization

Plot the K hyperplanes (lines) defined by W. Each row wⱼ defines a line orthogonal to wⱼ passing through -bⱼ/||wⱼ||² · ŵⱼ (if bias is included simply).

Overlay on data scatter plot. Do hyperplanes carve meaningful regions? Are many hyperplanes overlapping?

### Responsibility Map

Color each point by which unit claims it (max responsibility). Voronoi-like visualization. Are regions contiguous? Do they align with data structure?

### Activation Magnitude Map

Color each point by ||a||. Where does the layer produce strong activations? Weak activations?

### Weight Vector Plot

Plot weight vectors as arrows from origin. Do they spread out (diverse) or cluster (redundant)? For K > 2, project to 2D via PCA of W.

## Experimental Grid

Full factorial is expensive. Prioritize:

**Phase 1: Activation comparison without volume control**
- ReLU, softmax, log-softmax, z ⊙ softmax(z)
- K = 8, 16
- Gaussian clusters data
- 10 seeds each

Questions answered:
- Does competition reduce redundancy?
- Does collapse occur?
- What is reconstruction cost of competition?

**Phase 2: Volume control comparison on best competitive activation**
- Take most promising competitive activation from Phase 1
- Test all volume controls
- K = 16
- Gaussian clusters data
- 10 seeds each

Questions answered:
- Which volume control prevents collapse?
- Does volume control affect specialization?
- What is the regularization strength tradeoff?

**Phase 3: Data variation**
- Best activation + volume control from Phase 2
- Test on: clusters, rings, uniform, polytope
- K = 16
- 10 seeds each

Questions answered:
- Does competition help on non-mixture data?
- Is there data structure where competition hurts?

**Phase 4: Width scaling**
- Best configuration from Phase 3
- K = 4, 8, 16, 32
- Gaussian clusters data
- 10 seeds each

Questions answered:
- How does redundancy scale with width?
- Does competition maintain its advantage at larger K?

## Expected Outcomes

### Optimistic Scenario

Competitive activations show:
- Lower redundancy scores than ReLU
- Comparable reconstruction error
- Clean partitioning of input space
- No collapse with appropriate volume control

This would validate the core hypothesis: competition forces specialization without sacrificing performance.

### Pessimistic Scenario

Competitive activations show:
- Collapse without volume control
- Still-high redundancy even with volume control
- Significantly worse reconstruction error
- Unstable training dynamics

This would indicate that competition at hidden layers is fundamentally different from competition at output layers, and the implicit EM intuition doesn't transfer.

### Likely Scenario

Mixed results:
- Competition reduces redundancy somewhat
- Reconstruction error increases somewhat
- Some configurations collapse, others don't
- Volume control matters a lot

This would require careful analysis to understand the tradeoffs and identify which configurations are viable.

## Implementation Notes

### Measuring Angles in Higher Dimensions

For K > 2, weight vectors live in ℝ² (input dimension), not ℝᴷ. Pairwise angles are well-defined. The question is whether K vectors in ℝ² are diverse. With K > 2, perfect orthogonality is impossible—at most 2 orthogonal vectors in ℝ².

This is fine. We're not asking for orthogonality. We're asking whether vectors spread out to cover different directions, rather than clustering. In 2D, K = 8 vectors can be evenly spaced at 22.5° intervals. Redundancy would mean several vectors at nearly the same angle.

### Softmax Numerical Stability

For softmax, subtract max(z) before exponentiating:
```
r = softmax(z) = exp(z - max(z)) / Σ exp(z - max(z))
```

Prevents overflow. Standard practice.

### Log-Softmax Stability

PyTorch provides numerically stable log_softmax. Use it rather than log(softmax(z)).

### Detecting Collapse During Training

Add early stopping or alerts if:
- Any unit's average responsibility exceeds 0.9
- Any unit's average responsibility falls below 0.01
- Effective rank of W drops below 2

Don't stop the run—record that collapse occurred and when.

### Reproducibility

- Fix random seeds for data generation, weight initialization, and training
- Record all hyperparameters
- Save model checkpoints at regular intervals
- Log all metrics to file, not just final values

## What This Experiment Cannot Answer

- Whether competitive activations work in deep networks (need multi-layer experiments)
- Whether competitive activations help on real tasks (need task-specific experiments)
- The optimal competitive activation (too many candidates, need theory to narrow)
- Whether implicit EM interpretation is correct (this tests consequences, not mechanism)

This experiment tests one claim: competition reduces redundancy in a single layer. It's a necessary first step, not a complete validation.