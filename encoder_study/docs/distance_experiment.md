# Parameter Distance Experiment

## Motivation

This experiment investigates how the LSE+InfoMax objective affects the optimization trajectory of sparse autoencoders (SAEs). We measure the L2 distance in parameter space between initial and final states under different training regimes.

The core question: **Does LSE+InfoMax guide parameters to a qualitatively different region of parameter space, and does this region yield better representations?**

## Metrics

For each condition, we measure:

- **L2 Distance**: Euclidean distance between flattened parameter vectors
- **Cosine Similarity**: Angular similarity between parameter vectors
- **Reconstruction MSE**: Pixel-level reconstruction quality (primary metric)
- **Convergence EMA**: Exponential moving average of absolute loss deltas (lower = more converged)
- **L0 Sparsity**: Fraction of features active per input

## Experimental Conditions

### Condition 0: SAE Baseline

Standard SAE training from random initialization.

```
Random Init ──[SAE Training]──> Final
     │                            │
     └─────── distance ───────────┘
```

**Purpose**: Establish baseline distance and performance for comparison.

### Condition 1: Pretrain Encoder then SAE

First pretrain the encoder using LSE+InfoMax (unsupervised), then train the full SAE.

```
Random Init ──[LSE+InfoMax]──> Pretrained ──[SAE Training]──> Final
     │                              │                           │
     └──── distance A ──────────────┘                           │
                                    └──────── distance B ───────┘
     └────────────────────── distance C ────────────────────────┘
```

**Questions**:
- Does pretraining move parameters to a better starting point?
- Does subsequent SAE training preserve or undo the pretrained structure?
- Is the total distance (C) larger or smaller than baseline?

### Condition 2: SAE + LSE+InfoMax Regularizer

Train SAE from scratch with LSE+InfoMax added as a regularizer term.

```
Loss = SAE_Loss + λ * InfoMax_Loss
```

```
Random Init ──[SAE + InfoMax Reg]──> Final
     │                                 │
     └─────────── distance ────────────┘
```

**Questions**:
- Does the regularizer constrain the optimization trajectory?
- Does it improve representation quality without hurting reconstruction?

### Condition 3: Alternating Training Phases

Three-phase alternating training: SAE → Unsupervised → SAE

```
Random Init ──[SAE]──> Phase 1 ──[LSE+InfoMax]──> Phase 2 ──[SAE]──> Phase 3
     │                    │                          │                  │
     └── distance A ──────┘                          │                  │
                          └────── distance B ────────┘                  │
                                                     └── distance C ────┘
     └──────────────────────── total distance ──────────────────────────┘
```

**Questions**:
- Does unsupervised training (Phase 2) improve or disrupt SAE features?
- Can the model recover after unsupervised intervention?
- What is the net effect on representation quality?

## Hypotheses

1. **Pretraining hypothesis**: LSE+InfoMax pretraining moves parameters to a region that yields comparable or better reconstruction, even after SAE fine-tuning.

2. **Regularization hypothesis**: Adding LSE+InfoMax as a regularizer constrains the trajectory to stay closer to initialization while maintaining reconstruction quality.

3. **Alternating hypothesis**: Unsupervised LSE+InfoMax phases may disrupt reconstruction temporarily, but the model can recover with subsequent SAE training.

4. **Distance-quality correlation**: Conditions with larger parameter distance may correspond to either better or worse reconstruction, depending on whether the optimization found a good basin.

## Running the Experiment

```bash
cd encoder_study

# Run with default config
python scripts/run_distance_experiment.py

# Run with custom config
python scripts/run_distance_experiment.py --config config/distance_experiment.yaml

# Specify output directory
python scripts/run_distance_experiment.py --output-dir results/my_experiment

# Force CPU
python scripts/run_distance_experiment.py --device cpu
```

## Configuration

See `config/distance_experiment.yaml`:

```yaml
model:
  input_dim: 784
  hidden_dim: 64

sae:
  l1_weight: 0.01

infomax:
  lambda_lse: 1.0
  lambda_var: 1.0
  lambda_tc: 1.0

training:
  epochs: 100           # Total epochs for Conditions 0, 2
  pretrain_epochs: 50   # Pretraining epochs for Condition 1
  phase_epochs: 50      # Epochs per phase for Condition 3
  batch_size: 128
  lr: 0.001
  infomax_weight: 0.1   # Regularizer weight for Condition 2
  ema_decay: 0.9        # Decay for convergence EMA

seeds: [1, 2, 3]
```

## Output

Results are saved to `results/distance_experiment/distance_results.json`:

```json
[
  {
    "condition": 0,
    "seed": 1,
    "distance_init_final": 42.5,
    "cosine_init_final": 0.85,
    "convergence_ema": 0.00015,
    "metrics": {
      "reconstruction_mse": 0.026,
      "l0_sparsity": 32.2,
      "l0_density": 0.503
    }
  },
  ...
]
```

## Analysis Ideas

1. **Distance vs. Reconstruction**: Plot reconstruction MSE against parameter distance for each condition.

2. **Trajectory Visualization**: Use PCA or t-SNE on parameter snapshots to visualize optimization paths.

3. **Per-layer Analysis**: Compute distance separately for encoder and decoder to see which moves more.

4. **Phase Dynamics (Condition 3)**: Track how reconstruction MSE changes across phases to understand the interplay between supervised and unsupervised objectives.

5. **Ablation on Regularizer Weight**: Vary `infomax_weight` in Condition 2 to find the optimal balance between reconstruction and InfoMax objectives.
