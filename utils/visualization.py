"""
Load all experiment result JSONs from results/ and produce publication-quality plots:
  1. Dice convergence curves (all 4 experiments on one plot)
  2. Final Dice Score bar chart with per-class breakdown
"""
import os
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from config import RESULTS_DIR, NUM_ROUNDS

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "figure.dpi": 150,
})

COLORS = {
    "single_site":      "#E74C3C",
    "fedavg_iid":       "#3498DB",
    "fedavg_noniid":    "#2ECC71",
    "fedavg_noniid_dp": "#F39C12",
}
LABELS = {
    "single_site":      "Single-Site (Hospital A)",
    "fedavg_iid":       "FedAvg IID",
    "fedavg_noniid":    "FedAvg Non-IID",
    "fedavg_noniid_dp": "FedAvg Non-IID + DP (ε≈10)",
}


def load_results():
    data = {}

    # Single-site: epoch-level history
    path = os.path.join(RESULTS_DIR, "single_site_history.json")
    if os.path.exists(path):
        with open(path) as f:
            h = json.load(f)
        data["single_site"] = {
            "val_dice": h.get("val_dice", []),
            "test_dice": h.get("test_dice", 0),
        }

    # Federated experiments: round-level history
    for mode in ["iid", "noniid", "noniid_dp"]:
        key = f"fedavg_{mode}"
        path = os.path.join(RESULTS_DIR, f"{key}_history.json")
        if os.path.exists(path):
            with open(path) as f:
                h = json.load(f)
            dice_series = [entry["dice"] for entry in h]
            data[key] = {
                "val_dice": dice_series,
                "test_dice": dice_series[-1] if dice_series else 0,
            }

    return data


def plot_convergence(data):
    fig, ax = plt.subplots(figsize=(9, 5))

    for key, values in data.items():
        series = values["val_dice"]
        x = list(range(1, len(series) + 1))
        ax.plot(x, series, label=LABELS.get(key, key),
                color=COLORS.get(key, "grey"), linewidth=2, marker="o", markersize=3)

    ax.set_xlabel("Round / Epoch")
    ax.set_ylabel("Dice Score")
    ax.set_title("Federated vs. Single-Site Training Convergence\n(ACDC Cardiac Segmentation)")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = os.path.join(RESULTS_DIR, "convergence.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def plot_final_bar(data):
    keys   = [k for k in LABELS if k in data]
    labels = [LABELS[k] for k in keys]
    scores = [data[k]["test_dice"] for k in keys]
    colors = [COLORS[k] for k in keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, scores, color=colors, width=0.5, edgecolor="white", linewidth=1.2)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{score:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Dice Score")
    ax.set_title("Final Test Dice Score Comparison\n(ACDC Cardiac Segmentation)")
    ax.set_ylim(0, 1.05)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=scores[0], color=COLORS["single_site"],
               linestyle="--", linewidth=1, alpha=0.5, label="Single-site baseline")
    ax.legend()
    fig.tight_layout()

    out = os.path.join(RESULTS_DIR, "final_dice_bar.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def plot_dp_tradeoff(data):
    """Privacy-accuracy trade-off: compare Non-IID vs Non-IID + DP."""
    if "fedavg_noniid" not in data or "fedavg_noniid_dp" not in data:
        return

    no_dp = data["fedavg_noniid"]["test_dice"]
    with_dp = data["fedavg_noniid_dp"]["test_dice"]
    drop = (no_dp - with_dp) * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["No Privacy", "DP (ε≈10)"], [no_dp, with_dp],
           color=["#2ECC71", "#F39C12"], width=0.4, edgecolor="white")
    ax.set_ylabel("Dice Score")
    ax.set_title(f"Privacy-Accuracy Trade-off\n(Dice drop: {drop:.2f}%)")
    ax.set_ylim(0, 1.05)
    for i, v in enumerate([no_dp, with_dp]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = os.path.join(RESULTS_DIR, "dp_tradeoff.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    data = load_results()

    if not data:
        print("No results found. Run experiments first.")
        return

    plot_convergence(data)
    plot_final_bar(data)
    plot_dp_tradeoff(data)
    print("\nAll plots saved to results/")


if __name__ == "__main__":
    main()
