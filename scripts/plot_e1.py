"""Publication figures for Experiment E1 (planner comparison).

Reads results/raw/e1_summary.csv and results/raw/e1_steps.csv and writes three
figures to results/figures/:

    e1_coverage_over_time.png  — mean coverage vs step, per planner, per map
    e1_nodes_expanded.png      — mean nodes expanded per planner call, per map
    e1_planning_time.png       — distribution of mean planner wall-clock

Usage:  python -m scripts.plot_e1
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Okabe-Ito colourblind-safe palette, one colour per planner.
PLANNER_COLORS = {
    "astar": "#0072B2",  # blue
    "bfs": "#E69F00",    # orange
    "ucs": "#009E73",    # green
    "dfs": "#D55E00",    # vermillion
}
PLANNERS = ["astar", "bfs", "ucs", "dfs"]
MAPS = ["office", "maze", "open_field", "cluttered"]

FIG_DIR = Path("results/figures")
RAW = Path("results/raw")


def plot_coverage_over_time(steps: pd.DataFrame) -> None:
    """Mean coverage vs step for each planner, one subplot per map kind.

    Runs have different lengths, so each seed's coverage is interpolated onto a
    shared step grid before averaging (censored runs hold their final value).
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    for ax, map_kind in zip(axes.ravel(), MAPS):
        sub = steps[steps.map_kind == map_kind]
        max_step = int(sub.step.max()) if len(sub) else 1
        grid = np.linspace(1, max_step, 200)
        for planner in PLANNERS:
            curves = []
            psub = sub[sub.planner == planner]
            for _seed, g in psub.groupby("seed"):
                g = g.sort_values("step")
                curves.append(np.interp(grid, g.step.values, g.coverage.values,
                                        left=0.0, right=g.coverage.values[-1]))
            if not curves:
                continue
            mean = np.mean(curves, axis=0)
            ax.plot(grid, mean, color=PLANNER_COLORS[planner], label=planner, linewidth=2)
        ax.axhline(90, color="grey", linestyle=":", linewidth=1)
        ax.set_title(map_kind)
        ax.set_xlabel("simulation step")
        ax.set_ylabel("coverage (%)")
        ax.grid(alpha=0.3)
    axes.ravel()[0].legend(title="planner", loc="lower right")
    fig.suptitle("E1: coverage over time by planner (mean over 30 held-out seeds)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e1_coverage_over_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_nodes_expanded(summary: pd.DataFrame) -> None:
    """Grouped bar chart: mean nodes expanded per planner call, by map."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(MAPS))
    width = 0.2
    for i, planner in enumerate(PLANNERS):
        means = [summary[(summary.map_kind == m) & (summary.planner == planner)].nodes_mean.mean()
                 for m in MAPS]
        ax.bar(x + (i - 1.5) * width, means, width, label=planner, color=PLANNER_COLORS[planner])
    ax.set_xticks(x)
    ax.set_xticklabels(MAPS)
    ax.set_ylabel("mean nodes expanded per planner call")
    ax.set_title("E1: search effort per planner call (lower is better)")
    ax.legend(title="planner")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e1_nodes_expanded.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_planning_time(summary: pd.DataFrame) -> None:
    """Box plot of mean per-call planning wall-clock, by planner."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    data = [summary[summary.planner == p].plan_mean_ms.values for p in PLANNERS]
    bp = ax.boxplot(data, tick_labels=PLANNERS, patch_artist=True, showfliers=False)
    for patch, planner in zip(bp["boxes"], PLANNERS):
        patch.set_facecolor(PLANNER_COLORS[planner])
        patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set_color("black")
    ax.set_ylabel("mean planning time per call (ms)")
    ax.set_title("E1: planning wall-clock per call (over all maps x seeds)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e1_planning_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(RAW / "e1_summary.csv")
    steps = pd.read_csv(RAW / "e1_steps.csv")
    plot_coverage_over_time(steps)
    plot_nodes_expanded(summary)
    plot_planning_time(summary)
    print(f"wrote 3 figures to {FIG_DIR}/")


if __name__ == "__main__":
    main()
