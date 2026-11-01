import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Styling for publication-grade IEEE/ACM format
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 2,
    "lines.markersize": 7,
    "grid.alpha": 0.3
})

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_pareto_frontier():
    """Generates Figure 1: Utility vs. Worst-Case Counterfactual Shift Pareto Curve."""
    # Data from Experiment 2 run
    lambdas = [0.0, 0.05, 0.15, 0.30, 0.60, 1.20]
    auroc = [0.7295, 0.7349, 0.7161, 0.7285, 0.7328, 0.6295]
    worst_cf_shift = [0.1188, 0.1538, 0.1138, 0.1112, 0.0674, 0.0131]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    color_auroc = "#1f77b4"
    color_shift = "#d62728"

    ax1.set_xlabel(r"Fairness Regularization Strength ($\lambda$)")
    ax1.set_ylabel("Predictive Utility (AUROC)", color=color_auroc)
    line1 = ax1.plot(lambdas, auroc, marker="o", color=color_auroc, label="AUROC")
    ax1.tick_params(axis="y", labelcolor=color_auroc)
    ax1.set_ylim(0.58, 0.78)
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Worst-Case CF Shift ($\max_{G_k} \mathbb{E}[|\Delta f|]$)", color=color_shift)
    line2 = ax2.plot(lambdas, worst_cf_shift, marker="s", linestyle="--", color=color_shift, label="Worst CF Shift")
    ax2.tick_params(axis="y", labelcolor=color_shift)
    ax2.set_ylim(0.0, 0.20)

    # Highlight optimal operating point
    ax1.axvline(x=0.60, color="gray", linestyle=":", alpha=0.8)
    ax1.annotate("Optimal Operating Point\n($\lambda = 0.60$)", 
                 xy=(0.60, 0.7328), xytext=(0.70, 0.74),
                 arrowprops=dict(facecolor="black", arrowstyle="->", lw=1.2))

    plt.title("Pareto Trade-off: Utility Preservation vs. Worst-Case CF Sensitivity")
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "figure1_pareto_frontier.pdf")
    plt.savefig(fig_path, dpi=300)
    plt.savefig(fig_path.replace(".pdf", ".png"), dpi=300)
    print(f"Saved: {fig_path}")
    plt.close()


def plot_ablation_cross_graph():
    """Generates Figure 2: Cross-Graph Audit Discrepancy Across Ablated Strategies."""
    # Data from Experiment 3 run
    models = ["Unconstrained", "Single-Graph (G1)", "Average-Graph", "Minimax (Proposed)"]
    g1_shifts = [0.1661, 0.1218, 0.0739, 0.1073]
    g2_shifts = [0.1669, 0.1225, 0.0755, 0.1072]
    g3_shifts = [0.1632, 0.1193, 0.0729, 0.1045]
    g4_shifts = [0.1669, 0.1225, 0.0755, 0.1072]

    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x - 1.5 * width, g1_shifts, width, label="DAG G1 (Proxy Limit)", color="#4c72b0")
    ax.bar(x - 0.5 * width, g2_shifts, width, label="DAG G2 (Unbiased Alloc)", color="#55a868")
    ax.bar(x + 0.5 * width, g3_shifts, width, label="DAG G3 (Proxy Repay)", color="#c44e52")
    ax.bar(x + 1.5 * width, g4_shifts, width, label="DAG G4 (Mediated Edu)", color="#8172b3")

    ax.set_ylabel("Counterfactual Score Shift")
    ax.set_title("Audit Sensitivity Across Candidate DAGs by Defense Strategy")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(frameon=True)
    ax.grid(axis="y")

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "figure2_cross_graph_audit.pdf")
    plt.savefig(fig_path, dpi=300)
    plt.savefig(fig_path.replace(".pdf", ".png"), dpi=300)
    print(f"Saved: {fig_path}")
    plt.close()


def plot_simulation_thinning():
    """Generates Figure 3: Minority Thinning vs. Proxy Strength Interaction."""
    # Data from Experiment 4 run
    scenarios = ["Balanced\n(Mod Proxy)", "Balanced\n(Strong Proxy)", "Thinned\n(Mod Proxy)", "Thinned\n(Strong Proxy)"]
    baseline_gt = [0.0240, 0.0902, 0.0095, 0.0193]
    minimax_gt = [0.0439, 0.0600, 0.0066, 0.0122]

    x = np.arange(len(scenarios))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x - width/2, baseline_gt, width, label="Unconstrained Baseline", color="#dd8452")
    ax.bar(x + width/2, minimax_gt, width, label="Minimax Robust (Proposed)", color="#4c72b0")

    ax.set_ylabel("Ground-Truth Counterfactual Discrepancy")
    ax.set_title("Robustness Under Demographic Thinning & Proxy Strength")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend(frameon=True)
    ax.grid(axis="y")

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "figure3_simulation_thinning.pdf")
    plt.savefig(fig_path, dpi=300)
    plt.savefig(fig_path.replace(".pdf", ".png"), dpi=300)
    print(f"Saved: {fig_path}")
    plt.close()


if __name__ == "__main__":
    plot_pareto_frontier()
    plot_ablation_cross_graph()
    plot_simulation_thinning()
    print("\nAll publication-quality figures successfully generated in figures/")