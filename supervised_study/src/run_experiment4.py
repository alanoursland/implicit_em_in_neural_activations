"""Experiment 4: Why composition breaks EM conditioning.

Three measurements, each testing one claim from the composition analysis
(see notes/why_composition_breaks_em.md at repo root):

1. Gradient competition (--phase grad):
   Measure ||grad_d CE|| vs lambda*||grad_d aux|| at the intermediate layer
   during training, plus their cosine similarity. Report 3 asserted a ~1000:1
   ratio at lambda=0.001; this measures it.

2. Curvature tracking (--phase curvature):
   Power-iteration estimate of the spectral norm of the Hessian w.r.t. the
   distances d for (a) the LSE loss and (b) the CE loss through the head.
   Prediction: (a) stays below 0.5 (Bohning bound) at all times; (b) grows
   and tracks sigma_max(W2)^2.

3. Stop-gradient conditioning sweep (--phase sweep):
   Three arms x four SGD learning rates x seeds:
     - joint_lam_small: CE + 0.001*(LSE+var+tc), gradients flow everywhere
     - joint_lam_1:     CE + 1.0*(LSE+var+tc), gradients flow everywhere
     - stopgrad:        head reads y.detach(); W1 trained ONLY by LSE+var+tc
   Feature quality measured by a fixed-protocol linear probe on frozen
   distances (Adam, independent of the sweep lr). Prediction: probe accuracy
   and feature-health metrics are flat across lr for stopgrad (single-layer
   EM conditioning restored) and lr-sensitive for the joint arms.

Usage (from supervised_study/src/):
    python run_experiment4.py --phase sweep --data-dir <path>
    python run_experiment4.py --phase grad --data-dir <path>
    python run_experiment4.py --phase curvature --data-dir <path>
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import neg_log_softmin
from losses import variance_loss, correlation_loss
from metrics import dead_unit_count, min_variance, redundancy_score, responsibility_entropy
from utils import set_seed

_supervised_root = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Data: CPU-cached tensor loader (transforms applied once, not per epoch)
# ---------------------------------------------------------------------------

class CachedLoader:
    def __init__(self, data, labels, batch_size, shuffle=True):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        n = len(self.data)
        idx = torch.randperm(n) if self.shuffle else torch.arange(n)
        for i in range(0, n, self.batch_size):
            b = idx[i : i + self.batch_size]
            yield self.data[b], self.labels[b]

    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size


def load_mnist_cached(data_dir: str, batch_size: int = 128, device=None):
    from torchvision import datasets

    device = torch.device(device or "cpu")
    train = datasets.MNIST(root=data_dir, train=True, download=True)
    test = datasets.MNIST(root=data_dir, train=False, download=True)
    train_x = train.data.float().div_(255.0).view(len(train), -1).to(device)
    train_y = train.targets.to(device)
    test_x = test.data.float().div_(255.0).view(len(test), -1).to(device)
    test_y = test.targets.to(device)
    return (
        CachedLoader(train_x, train_y, batch_size, shuffle=True),
        train_x, train_y, test_x, test_y,
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Exp4Model(nn.Module):
    """Two-layer model with optional stop-gradient between EM layer and head.

    x -> Linear -> ReLU -> d -> NegLogSoftmin -> [detach?] -> Linear -> LayerNorm -> logits

    The auxiliary loss (LSE + var + tc) is always applied to raw distances d.
    With detach_head=True the CE gradient cannot reach W1: the EM layer is
    trained purely by the auxiliary objective (Paper 2's setting), while the
    head trains on the frozen-at-each-step representation.
    """

    def __init__(self, input_dim=784, hidden_dim=25, num_classes=10,
                 use_nls=True, detach_head=False):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.layer_norm = nn.LayerNorm(num_classes)
        self.use_nls = use_nls
        self.detach_head = detach_head

    def forward(self, x):
        d = self.act(self.linear(x))
        y = neg_log_softmin(d) if self.use_nls else d
        y_head = y.detach() if self.detach_head else y
        h = self.layer_norm(self.classifier(y_head))
        return h, d


def aux_loss(d, variance_eps=1e-6):
    """LSE (mean over batch) + variance + decorrelation on raw distances."""
    lse = -torch.logsumexp(-d, dim=1).mean()
    var = variance_loss(d, eps=variance_eps)
    tc = correlation_loss(d)
    return lse + var + tc, {"lse": lse.item(), "var": var.item(), "tc": tc.item()}


# ---------------------------------------------------------------------------
# Linear probe (fixed protocol, isolates feature quality from head training)
# ---------------------------------------------------------------------------

def linear_probe_accuracy(model, train_x, train_y, test_x, test_y,
                          epochs=15, batch_size=256, lr=1e-3, seed=0):
    model.eval()
    with torch.no_grad():
        feat_train = model.act(model.linear(train_x))
        feat_test = model.act(model.linear(test_x))

    device = feat_train.device
    g = torch.Generator(device=device).manual_seed(seed)
    probe = nn.Linear(feat_train.shape[1], 10).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    n = len(feat_train)
    for _ in range(epochs):
        idx = torch.randperm(n, generator=g, device=device)
        for i in range(0, n, batch_size):
            b = idx[i : i + batch_size]
            opt.zero_grad()
            loss = F.cross_entropy(probe(feat_train[b]), train_y[b])
            loss.backward()
            opt.step()
    with torch.no_grad():
        acc = (probe(feat_test).argmax(1) == test_y).float().mean().item()
    return acc


# ---------------------------------------------------------------------------
# Shared training loop
# ---------------------------------------------------------------------------

def train_run(arm, lr, seed, loaders, epochs=40, hidden_dim=25,
              optimizer_name="sgd", instrument=None, log_interval=10):
    """Train one run. arm in {joint_lam_small, joint_lam_1, stopgrad}.

    instrument: optional dict {"grad_every": int, "curvature_every": int,
    "probe_batch": tensor} enabling in-training measurements.
    """
    train_loader, train_x, train_y, test_x, test_y = loaders
    set_seed(seed)

    detach = arm == "stopgrad"
    lam = {"joint_lam_small": 0.001, "joint_lam_mid": 0.03,
           "joint_lam_1": 1.0, "stopgrad": 1.0}[arm]

    device = train_x.device
    model = Exp4Model(hidden_dim=hidden_dim, detach_head=detach).to(device)
    if optimizer_name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)

    curves = {k: [] for k in [
        "epoch", "ce", "aux", "test_acc", "dead_units", "min_var",
        "redundancy", "resp_entropy", "aux_lse_test",
    ]}
    grad_log = {k: [] for k in ["step", "g_ce", "g_aux", "cos"]}
    curv_log = {k: [] for k in ["epoch", "h_ce", "h_lse", "sigma_max_w2"]}

    step = 0
    for epoch in range(epochs):
        model.train()
        ep_ce, ep_aux, nb = 0.0, 0.0, 0
        for x, labels in train_loader:
            opt.zero_grad()
            h, d = model(x)
            ce = F.cross_entropy(h, labels)
            aux, _ = aux_loss(d)
            total = ce + lam * aux

            if instrument and instrument.get("grad_every") and step % instrument["grad_every"] == 0:
                g_ce = torch.autograd.grad(ce, d, retain_graph=True)[0] if not detach \
                    else torch.zeros_like(d)
                g_aux = torch.autograd.grad(lam * aux, d, retain_graph=True)[0]
                n_ce, n_aux = g_ce.norm().item(), g_aux.norm().item()
                cos = float(F.cosine_similarity(
                    g_ce.flatten(), g_aux.flatten(), dim=0).item()) \
                    if n_ce > 0 and n_aux > 0 else float("nan")
                grad_log["step"].append(step)
                grad_log["g_ce"].append(n_ce)
                grad_log["g_aux"].append(n_aux)
                grad_log["cos"].append(cos)

            total.backward()
            opt.step()
            ep_ce += ce.item()
            ep_aux += aux.item()
            nb += 1
            step += 1

        model.eval()
        with torch.no_grad():
            h_t, d_t = model(test_x)
            acc = (h_t.argmax(1) == test_y).float().mean().item()
            r = torch.softmax(-d_t, dim=1)
            lse_test = -torch.logsumexp(-d_t, dim=1).mean().item()
        curves["epoch"].append(epoch + 1)
        curves["ce"].append(ep_ce / nb)
        curves["aux"].append(ep_aux / nb)
        curves["test_acc"].append(acc)
        curves["dead_units"].append(dead_unit_count(d_t))
        curves["min_var"].append(min_variance(d_t))
        curves["redundancy"].append(redundancy_score(d_t))
        curves["resp_entropy"].append(responsibility_entropy(r))
        curves["aux_lse_test"].append(lse_test)

        if instrument and instrument.get("curvature_every") and \
                (epoch % instrument["curvature_every"] == 0 or epoch == epochs - 1):
            hc, hl, sw = curvature_probe(model, instrument["probe_batch"])
            curv_log["epoch"].append(epoch + 1)
            curv_log["h_ce"].append(hc)
            curv_log["h_lse"].append(hl)
            curv_log["sigma_max_w2"].append(sw)

        if (epoch + 1) % log_interval == 0 or epoch == 0:
            print(f"  [{arm} lr={lr} seed={seed}] ep {epoch+1:3d} "
                  f"CE={ep_ce/nb:.4f} aux={ep_aux/nb:.3f} acc={acc:.4f} "
                  f"dead={curves['dead_units'][-1]}", flush=True)

    probe_acc = linear_probe_accuracy(model, train_x, train_y, test_x, test_y)
    return model, curves, grad_log, curv_log, probe_acc


# ---------------------------------------------------------------------------
# Curvature probe: Hessian spectral norm w.r.t. d via power iteration
# ---------------------------------------------------------------------------

def _lam_max(f_builder, d0, iters=30, seed=0):
    d = d0.detach().requires_grad_(True)
    f = f_builder(d)
    g = torch.autograd.grad(f, d, create_graph=True)[0]
    gen = torch.Generator(device=d.device).manual_seed(seed)
    v = torch.randn(d.shape, generator=gen, device=d.device)
    v = v / v.norm()
    lam = 0.0
    for _ in range(iters):
        hv = torch.autograd.grad(g, d, grad_outputs=v, retain_graph=True)[0]
        lam = hv.norm().item()
        if lam < 1e-12:
            return 0.0
        v = hv / hv.norm()
    return lam


def curvature_probe(model, probe_batch):
    """Spectral norms of Hessians w.r.t. d on a fixed probe batch.

    Both losses use SUM reduction so the Hessians are block-diagonal over
    samples and the spectral norm equals the worst per-sample curvature
    (LayerNorm and NLS both act per-sample). True labels are used: the
    Hessian of CE w.r.t. logits is label-free, but the second-order terms
    through NLS and LayerNorm enter via (p - y), which is not.
    """
    x_probe, y_probe = probe_batch
    model.eval()
    with torch.no_grad():
        d0 = model.act(model.linear(x_probe))

    def ce_from_d(d):
        y = neg_log_softmin(d) if model.use_nls else d
        h = model.layer_norm(model.classifier(y))
        return F.cross_entropy(h, y_probe, reduction="sum")

    def lse_from_d(d):
        return -torch.logsumexp(-d, dim=1).sum()

    h_ce = _lam_max(ce_from_d, d0)
    h_lse = _lam_max(lse_from_d, d0)
    sigma_w2 = torch.linalg.matrix_norm(model.classifier.weight, ord=2).item()
    return h_ce, h_lse, sigma_w2


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def tail_mean(vals, k=5):
    return sum(vals[-k:]) / min(k, len(vals))


def _load_json_if_present(path, default):
    if not path.exists():
        return default
    with open(path) as f:
        return json.load(f)


def _sweep_key(arm, lr, seed):
    return arm, float(lr), int(seed)


def phase_sweep(loaders, out_dir, epochs, seeds, hidden_dim, arms=None):
    arms = arms or ["joint_lam_small", "joint_lam_1", "stopgrad"]
    lrs = [0.0001, 0.001, 0.01, 0.1]
    results_path = out_dir / "sweep_results.json"
    curves_path = out_dir / "sweep_curves.json"
    results = _load_json_if_present(results_path, [])
    curves_all = _load_json_if_present(curves_path, {})
    completed = {_sweep_key(r["arm"], r["lr"], r["seed"]) for r in results}

    for arm in arms:
        for lr in lrs:
            for seed in seeds:
                key = _sweep_key(arm, lr, seed)
                if key in completed:
                    print(f"SKIP {arm} lr={lr} seed={seed} (already in {results_path.name})",
                          flush=True)
                    continue

                _, curves, _, _, probe_acc = train_run(
                    arm, lr, seed, loaders, epochs=epochs, hidden_dim=hidden_dim)
                rec = {
                    "arm": arm, "lr": lr, "seed": seed,
                    "probe_acc": probe_acc,
                    "test_acc": tail_mean(curves["test_acc"]),
                    "dead_units": tail_mean(curves["dead_units"]),
                    "min_var": tail_mean(curves["min_var"]),
                    "redundancy": tail_mean(curves["redundancy"]),
                    "resp_entropy": tail_mean(curves["resp_entropy"]),
                    "aux_lse_test": tail_mean(curves["aux_lse_test"]),
                }
                results.append(rec)
                curves_all[f"{arm}_lr{lr}_s{seed}"] = curves
                print(f"DONE {arm} lr={lr} seed={seed} probe={probe_acc:.4f} "
                      f"head={rec['test_acc']:.4f} dead={rec['dead_units']:.1f}",
                      flush=True)
                with open(results_path, "w") as f:
                    json.dump(results, f, indent=2)
                with open(curves_path, "w") as f:
                    json.dump(curves_all, f)
                completed.add(key)

    with open(curves_path, "w") as f:
        json.dump(curves_all, f)
    print("SWEEP COMPLETE", flush=True)


def phase_grad(loaders, out_dir, epochs, seeds, hidden_dim):
    """Gradient competition in the joint arms, Adam lr=1e-3 (report 3's operating point)."""
    out = {}
    for arm in ["joint_lam_small", "joint_lam_1"]:
        for seed in seeds:
            _, curves, grad_log, _, _ = train_run(
                arm, 0.001, seed, loaders, epochs=epochs, hidden_dim=hidden_dim,
                optimizer_name="adam",
                instrument={"grad_every": 100})
            out[f"{arm}_s{seed}"] = {"grad": grad_log,
                                     "test_acc": tail_mean(curves["test_acc"])}
            with open(out_dir / "grad_results.json", "w") as f:
                json.dump(out, f)
    print("GRAD COMPLETE", flush=True)


def phase_curvature(loaders, out_dir, epochs, seeds, hidden_dim):
    """Curvature tracking in joint (lam=0.001, Adam lr=1e-3) and stopgrad arms."""
    _, train_x, train_y, _, _ = loaders
    probe_batch = (train_x[:512], train_y[:512])
    out = {}
    for arm, opt_name in [("joint_lam_small", "adam"), ("stopgrad", "adam")]:
        for seed in seeds:
            _, curves, _, curv_log, _ = train_run(
                arm, 0.001, seed, loaders, epochs=epochs, hidden_dim=hidden_dim,
                optimizer_name=opt_name,
                instrument={"curvature_every": 2, "probe_batch": probe_batch})
            out[f"{arm}_s{seed}"] = {"curv": curv_log,
                                     "test_acc": tail_mean(curves["test_acc"])}
            with open(out_dir / "curvature_results.json", "w") as f:
                json.dump(out, f)
    print("CURVATURE COMPLETE", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["sweep", "grad", "curvature"], required=True)
    p.add_argument("--arms", nargs="+", default=None,
                   help="Sweep arms (default: joint_lam_small joint_lam_1 stopgrad)")
    p.add_argument("--data-dir", type=str, default="E:/ml_datasets")
    p.add_argument("--output-dir", type=str,
                   default=str(_supervised_root / "results" / "experiment4"))
    p.add_argument("--device", type=str, default="auto",
                   help="Device to use: auto, cpu, cuda, or cuda:N")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--hidden-dim", type=int, default=25)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    print(f"Using device: {device}", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    loaders = load_mnist_cached(args.data_dir, batch_size=args.batch_size, device=device)
    if args.phase == "sweep":
        phase_sweep(loaders, out_dir, args.epochs, args.seeds, args.hidden_dim,
                    arms=args.arms)
    elif args.phase == "grad":
        phase_grad(loaders, out_dir, args.epochs, args.seeds, args.hidden_dim)
    else:
        phase_curvature(loaders, out_dir, args.epochs, args.seeds, args.hidden_dim)


if __name__ == "__main__":
    main()
