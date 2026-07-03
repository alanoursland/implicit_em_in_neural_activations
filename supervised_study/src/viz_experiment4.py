"""Figures for Experiment 4 (composition breaks EM conditioning).

Reads results/experiment4 (+ experiment4_mid if present) and writes figures
into reports/figures/ so the report renders on GitHub.

Usage (from supervised_study/src/):
    python viz_experiment4.py
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_root = Path(__file__).resolve().parent.parent
RES = _root / "results" / "experiment4"
RES_MID = _root / "results" / "experiment4_mid"
FIG = _root / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

ARM_LABELS = {
    "joint_lam_small": "joint, λ=0.001 (CE-dominated)",
    "joint_lam_mid": "joint, λ=0.03 (balanced)",
    "joint_lam_1": "joint, λ=1 (EM-dominated)",
    "stopgrad": "stop-gradient (pure EM at W₁)",
}
ARM_COLORS = {
    "joint_lam_small": "#d62728",
    "joint_lam_mid": "#ff7f0e",
    "joint_lam_1": "#1f77b4",
    "stopgrad": "#2ca02c",
}
ARM_ORDER = ["joint_lam_small", "joint_lam_mid", "joint_lam_1", "stopgrad"]


def load_sweep():
    rows = json.load(open(RES / "sweep_results.json"))
    mid = RES_MID / "sweep_results.json"
    if mid.exists():
        rows += json.load(open(mid))
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        agg[r["arm"]][r["lr"]].append(r)
    return agg


def fig_conditioning(agg):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for metric, ax, ylabel in [
        ("probe_acc", axes[0], "Linear probe accuracy on frozen d"),
        ("redundancy", axes[1], "Redundancy ‖Corr(d)−I‖²_F (off-diag)"),
    ]:
        for arm in ARM_ORDER:
            if arm not in agg:
                continue
            lrs = sorted(agg[arm])
            mean = [np.mean([r[metric] for r in agg[arm][lr]]) for lr in lrs]
            std = [np.std([r[metric] for r in agg[arm][lr]]) for lr in lrs]
            ax.errorbar(lrs, mean, yerr=std, marker="o", capsize=3,
                        label=ARM_LABELS[arm], color=ARM_COLORS[arm])
        ax.set_xscale("log")
        ax.set_xlabel("SGD learning rate")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Feature quality vs learning rate: lr-sensitivity follows CE dominance")
    fig.tight_layout()
    fig.savefig(FIG / "exp4_conditioning.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_gradient_competition():
    data = json.load(open(RES / "grad_results.json"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    for key, color, label in [
        ("joint_lam_small_s42", "#d62728", "λ=0.001"),
        ("joint_lam_1_s42", "#1f77b4", "λ=1"),
    ]:
        g = data[key]["grad"]
        steps = np.array(g["step"])
        ax.plot(steps, g["g_ce"], color=color, label=f"‖∇_d CE‖  ({label})")
        ax.plot(steps, g["g_aux"], color=color, linestyle="--",
                label=f"λ‖∇_d aux‖  ({label})")
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("gradient norm at intermediate layer")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("Gradient magnitudes (seed 42)")

    ax = axes[1]
    for key in data:
        g = data[key]["grad"]
        color = "#d62728" if "small" in key else "#1f77b4"
        ratio = np.array(g["g_ce"]) / np.maximum(np.array(g["g_aux"]), 1e-12)
        ax.plot(g["step"], ratio, color=color, alpha=0.6)
    ax.axhline(1.0, color="k", linewidth=0.8, linestyle=":")
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("‖∇_d CE‖ / λ‖∇_d aux‖")
    ax.grid(alpha=0.3)
    ax.set_title("Dominance ratio (all seeds; red λ=0.001, blue λ=1)")

    fig.tight_layout()
    fig.savefig(FIG / "exp4_gradient_competition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_curvature():
    data = json.load(open(RES / "curvature_results.json"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    for key, v in data.items():
        c = v["curv"]
        joint = "joint" in key
        color = "#d62728" if joint else "#2ca02c"
        ax.plot(c["epoch"], c["h_ce"], color=color, alpha=0.7,
                label="‖∇²_d CE‖ (joint λ=0.001)" if key.endswith("s42") and joint else
                      ("‖∇²_d CE‖ (stop-grad)" if key.endswith("s42") else None))
        ax.plot(c["epoch"], c["h_lse"], color=color, alpha=0.7, linestyle="--",
                label="‖∇²_d LSE‖ (joint λ=0.001)" if key.endswith("s42") and joint else
                      ("‖∇²_d LSE‖ (stop-grad)" if key.endswith("s42") else None))
    ax.axhline(0.5, color="k", linewidth=1.0, linestyle=":", label="Böhning bound ½")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Hessian spectral norm w.r.t. d (sum reduction, batch 512)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_title("LSE curvature obeys the ½ bound; CE-path curvature does not")

    ax = axes[1]
    for key, v in data.items():
        if "joint" not in key:
            continue
        c = v["curv"]
        ax.plot(c["epoch"], np.array(c["sigma_max_w2"]) ** 2,
                color="#7f7f7f", linestyle="--",
                label="σ_max(W₂)²" if key.endswith("s42") else None)
        ax.plot(c["epoch"], c["h_ce"], color="#d62728", alpha=0.7,
                label="‖∇²_d CE‖" if key.endswith("s42") else None)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("spectral norm")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("Joint arm: CE-path curvature is parameter-dependent")

    fig.tight_layout()
    fig.savefig(FIG / "exp4_curvature.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    agg = load_sweep()
    fig_conditioning(agg)
    fig_gradient_competition()
    fig_curvature()
    print(f"Figures written to {FIG}")
