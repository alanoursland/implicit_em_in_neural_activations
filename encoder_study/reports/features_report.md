# Experiment 5: Feature Visualization

## Objective

Visualize the learned weight vectors to understand what the decoder-free encoder has learned.

## Method

Each row of the encoder weight matrix W is a 784-dimensional vector. Reshape to 28×28 and display as an image. Use a diverging colormap (RdBu): blue = positive weights, red = negative weights, white = zero.

```python
W = model.encoder.weight.data  # (64, 784)
features = W.reshape(64, 28, 28)
```

## Result

![Learned Features](../figures/fig5_features.png)

## Interpretation

### These are Prototypes, Not Parts

Standard sparse autoencoders and dictionary learning methods typically learn *parts*—edges, strokes, curves, Gabor-like filters that combine additively to reconstruct inputs.

These features are *prototypes*—whole digit templates that compete to explain inputs. Clear examples:

- Multiple 0s (circular templates)
- Multiple 1s (vertical strokes, various slants)
- Multiple 3s, 5s, 7s, 8s, 9s
- Style variants (loopy 2s, angular 2s)

This is mixture model behavior, not sparse coding behavior. In a Gaussian Mixture Model, each component centroid ends up looking like an exemplar of the data it claims. That's exactly what we see here.

### Center-Surround Structure

Most features show a clear digit shape (blue) surrounded by a halo of opposite sign (red). This center-surround structure means each feature is:

- A detector: "I fire when I see this digit"
- An anti-detector: "I suppress when I see other digits"

This is discriminative structure, not reconstructive structure. The model isn't trying to reproduce pixels—it's trying to claim inputs competitively.

### No Dead Units

All 64 features show distinct, interpretable structure. No grey noise. No degenerate patterns. Every component learned something.

This confirms the ablation results: the variance penalty prevents collapse, the decorrelation penalty prevents redundancy, and the LSE term creates competition. All 64 components found their niche.

### Unsupervised Discovery

These prototypes emerged from:

- Raw pixels (no preprocessing)
- No labels (unsupervised)
- No reconstruction target (no decoder)
- No contrastive pairs (no augmentation)
- No explicit sparsity penalty (no L1)

The only signal was: "Compete softly (LSE), stay alive (variance), be different (decorrelation)."

And the model discovered digits.

Not edges. Not strokes. Not generic Gabor filters. The actual latent categories of MNIST—with style variants.

## Connection to EM

This visualization is direct evidence for the implicit EM interpretation.

When you run EM on a Gaussian Mixture Model with unlabeled data, you say "find K clusters" and it recovers the underlying categories. The centroids end up looking like exemplars of the classes—even though you never told it there were classes.

This model did the same thing. 64 components, soft competition via responsibilities, and it converged on "these are the ~10 categories of things in this dataset, with multiple style variants per category."

## Connection to Probe Accuracy

This explains why the decoder-free encoder beats the standard SAE on linear probe accuracy (93.4% vs 90.3%).

| Model | What it learns | Probe task |
|-------|---------------|------------|
| Standard SAE | Parts (edges, strokes) | Compose parts into digit prediction |
| Ours | Prototypes (digit templates) | Map prototype to digit label |

A linear probe on prototypes barely has to do anything. Feature 23 fires → class 7. Feature 41 fires → class 0. The hard work is already done.

A linear probe on parts must learn to compose them. "Vertical stroke + top curve + bottom curve = 5." That's harder.

## This is a Neural Mixture Model

The features reveal that this isn't really a sparse autoencoder at all. It's a neural mixture model:

| GMM | This Model |
|-----|------------|
| Centroids μ_k | Rows of W |
| Soft assignment P(k\|x) | Responsibilities r = softmax(-Wx) |
| Compete for data | LSE term |
| Don't collapse | Variance term = log-det diagonal |
| Stay different | Decorrelation term = log-det off-diagonal |

The visualization makes the theory concrete. The learned representations have the structure you'd expect from mixture modeling, not from sparse coding.

## Summary

The features provide visual evidence for every claim in the paper:

| Claim | Visual Evidence |
|-------|-----------------|
| Implicit EM | Prototypes, not parts |
| Soft competition | Digit templates compete for inputs |
| Volume control works | 64 distinct, non-redundant features |
| Mixture model behavior | Centroids look like class exemplars |
| Unsupervised discovery | Found digits without labels |

## Files

- Figure: `figures/fig5_features.png`
- Checkpoint: `results/benchmark/ours_seed1.pt`

## Code

```bash
python scripts/visualize_features.py --checkpoint results/benchmark/ours_seed1.pt --output figures/fig5_features.png
```