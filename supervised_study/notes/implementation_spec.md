# Implementation Spec

## Purpose

This document bridges the experiment design to the existing codebase. It tells an implementer exactly what to build, what to reuse, and what to output.

## Existing Code to Reuse

The following modules exist and should be reused without modification:

### losses.py

- `lse_loss(a)` → returns (loss, responsibilities). Loss is `-logsumexp(-a).sum()`. Responsibilities are `softmax(-a)`.
- `variance_loss(a, eps)` → returns `-log(Var(a_j) + eps).sum()` over features.
- `correlation_loss(a)` → returns `||Corr(A) - I||²_F` off-diagonal only.
- `combined_loss(a, W, config)` → combines LSE + var + tc + wr with configurable weights. Config keys: `lambda_lse`, `lambda_var`, `lambda_tc`, `lambda_wr`, `variance_eps`.

### metrics.py

- `dead_units(a, threshold=0.01)` → count of features with variance < threshold.
- `redundancy_score(a)` → `||Corr(A) - I||²_F` off-diagonal.
- `responsibility_entropy(r)` → mean `H(r)` per sample.
- `sparsity_l0(a, threshold=0.01)` → (l0, density) tuple.
- `linear_probe_accuracy(train_features, train_labels, test_features, test_labels)` → float. Uses sklearn LogisticRegression.

### data.py

- `get_mnist(batch_size, flatten, data_dir, use_gpu_cache, device)` → (train_loader, test_loader).
- **Note for Claude Code environment:** Set `data_dir="./data"` and `use_gpu_cache=False` (or `True` with `device="cuda"` if GPU available). The default `E:/ml_datasets` is a Windows path that won't work.
- The GPU-cached loader returns `(data, labels)` tuples. The standard loader returns `(data, labels)` tuples from MNIST dataset. Both formats work the same way.

## New Code to Write

### NegLogSoftmin

One function:

```python
def neg_log_softmin(d):
    """Calibrate distances: y_j = d_j + log Z where Z = sum exp(-d_k).
    
    Args:
        d: distances (batch, K), non-negative
    Returns:
        y: calibrated distances (batch, K)
    """
    return d + torch.logsumexp(-d, dim=1, keepdim=True)
```

### SupervisedModel

One model class with configuration flags:

```python
class SupervisedModel(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=64, num_classes=10, 
                 use_neg_log_softmin=False):
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.softplus = nn.Softplus()
        self.use_neg_log_softmin = use_neg_log_softmin
        self.layer2 = nn.Linear(hidden_dim, num_classes)
        self.layer_norm = nn.LayerNorm(num_classes)
    
    def forward(self, x):
        d = self.softplus(self.layer1(x))       # raw distances (batch, K)
        if self.use_neg_log_softmin:
            y = neg_log_softmin(d)               # calibrated distances
        else:
            y = d                                # pass through
        h = self.layer_norm(self.layer2(y))      # class logits
        return h, d    # h for CE loss, d for auxiliary loss
    
    @property
    def W1(self):
        return self.layer1.weight                # (K, 784) for visualization
```

Returns both `h` (for cross-entropy) and `d` (for auxiliary loss). The auxiliary loss operates on `d` regardless of whether NegLogSoftmin is used in the forward pass.

### Config Mapping

Six configs map to model flags + loss weights:

```python
CONFIGS = {
    "baseline": {
        "use_neg_log_softmin": False,
        "loss": {"lambda_lse": 0.0, "lambda_var": 0.0, "lambda_tc": 0.0}
    },
    "nls_only": {
        "use_neg_log_softmin": True,
        "loss": {"lambda_lse": 0.0, "lambda_var": 0.0, "lambda_tc": 0.0}
    },
    "nls_lse": {
        "use_neg_log_softmin": True,
        "loss": {"lambda_lse": 1.0, "lambda_var": 0.0, "lambda_tc": 0.0}
    },
    "nls_lse_var": {
        "use_neg_log_softmin": True,
        "loss": {"lambda_lse": 1.0, "lambda_var": 1.0, "lambda_tc": 0.0}
    },
    "full_implicit_em": {
        "use_neg_log_softmin": True,
        "loss": {"lambda_lse": 1.0, "lambda_var": 1.0, "lambda_tc": 1.0}
    },
    "var_tc_only": {
        "use_neg_log_softmin": True,
        "loss": {"lambda_lse": 0.0, "lambda_var": 1.0, "lambda_tc": 1.0}
    },
}
```

Note: `lambda_wr` is always 0.0 (weight redundancy loss not used, same as Paper 2).

### Training Loop

For each config, for each seed:

```python
def train_one(config_name, seed):
    set_seed(seed)
    
    cfg = CONFIGS[config_name]
    model = SupervisedModel(use_neg_log_softmin=cfg["use_neg_log_softmin"])
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    loss_config = cfg["loss"]
    lambda_aux = 1.0  # overall weight for auxiliary loss
    
    for epoch in range(100):
        model.train()
        for x, labels in train_loader:
            optimizer.zero_grad()
            
            h, d = model(x)
            
            # Supervised loss
            ce_loss = F.cross_entropy(h, labels)
            
            # Auxiliary loss (volume control on raw distances)
            aux = combined_loss(d, model.W1, loss_config)
            aux_loss = aux["total"]
            
            # Total
            total = ce_loss + lambda_aux * aux_loss
            total.backward()
            optimizer.step()
    
    return model
```

**Important:** When all lambda values in loss_config are 0.0 (configs "baseline" and "nls_only"), `combined_loss` returns a total of 0.0. The auxiliary term contributes nothing. This is correct — those configs have no auxiliary loss.

### Evaluation

After training, evaluate on full test set:

```python
def evaluate_model(model, test_loader, loss_config):
    model.eval()
    all_d = []
    all_h = []
    all_labels = []
    
    with torch.no_grad():
        for x, labels in test_loader:
            h, d = model(x)
            all_d.append(d)
            all_h.append(h)
            all_labels.append(labels)
    
    d = torch.cat(all_d)
    h = torch.cat(all_h)
    labels = torch.cat(all_labels)
    
    # Classification accuracy
    preds = h.argmax(dim=1)
    accuracy = (preds == labels).float().mean().item()
    
    # Intermediate layer metrics (on distances d)
    dead = dead_units(d, threshold=0.01)
    redundancy = redundancy_score(d)
    
    # Responsibilities from distances
    r = torch.softmax(-d, dim=1)
    resp_ent = responsibility_entropy(r)
    
    return {
        "dead_units": dead,
        "redundancy": redundancy,
        "resp_entropy": resp_ent,
        "accuracy": accuracy,
    }
```

### Weight Visualization

One figure with six panels (one per config). Each panel shows a grid of 64 weight images (8×8 grid of 28×28 images).

```python
def visualize_weights(model, title):
    W = model.W1.detach().cpu()  # (64, 784)
    # Reshape each row to 28x28
    # Display as 8x8 grid
    # Diverging colormap (blue positive, red negative, white zero)
    # Scale to model's own max absolute weight
```

Same format as Paper 2 Figure 2. Use matplotlib. Save as PNG.

## Output

### Primary: Ablation table

Print and save as CSV:

```
Config,Dead Units,Redundancy,Resp Entropy,Accuracy
baseline,X±X,X±X,X±X,X±X
nls_only,...
nls_lse,...
nls_lse_var,...
full_implicit_em,...
var_tc_only,...
```

Mean ± std across 3 seeds.

### Secondary: Weight visualization

One PNG per config, or one combined figure with six panels. Saved to output directory.

### Logging

Print per-epoch: epoch, CE loss, auxiliary loss, total loss. Enough to diagnose training issues but not the primary output.

## Environment

- PyTorch (any recent version)
- torchvision (for MNIST)
- scikit-learn (for linear_probe_accuracy, if used)
- matplotlib (for visualization)
- numpy

No GPU required (MNIST is small). GPU helps but is not necessary. If no GPU, set `device="cpu"` and `use_gpu_cache=False` in `get_mnist`.

## Run Order

```
for config_name in CONFIGS:
    for seed in [42, 43, 44]:
        model = train_one(config_name, seed)
        results = evaluate_model(model, test_loader)
        save(results)
        visualize_weights(model, config_name)

aggregate_and_print_table()
```

18 runs total. Each run: ~60K samples × 100 epochs × forward+backward. On CPU, ~5-10 minutes per run. On GPU, ~1-2 minutes per run. Total: 1.5-3 hours CPU, 20-40 minutes GPU.