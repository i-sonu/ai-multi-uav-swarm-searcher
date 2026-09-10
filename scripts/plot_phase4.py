"""Publication figures for the Phase 4 experiments E2, E3, E4.

Reads results/raw/e{2,3,4}_summary.csv (+ e3_steps.csv) and writes figures to
results/figures/. Each experiment is plotted only if its CSV exists, so this can
be run after any subset of the sweeps.

    python -m scripts.plot_phase4
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MAPS = ["office", "maze", "open_field", "cluttered"]
METHOD_COLORS = {"none": "#D55E00", "greedy": "#E69F00", "hungarian": "#0072B2"}
N_COLORS = {1: "#999999", 2: "#0072B2", 3: "#009E73"}

FIG_DIR = Path("results/figures")
RAW = Path("results/raw")


def _mean_ttc(sub: pd.DataFrame) -> float:
    """Mean time-to-90% over the runs that reached it (censored runs excluded)."""
    vals = pd.to_numeric(sub["time_to_90"], errors="coerce").dropna()
    return float(vals.mean()) if len(vals) else float("nan")


# --------------------------------------------------------------------------- #
# E2 — allocation ablation
# --------------------------------------------------------------------------- #
def plot_e2(summary: pd.DataFrame) -> None:
    methods = ["none", "greedy", "hungarian"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(len(MAPS))
    width = 0.25
    # (a) redundant-coverage ratio — the key coordination metric (lower = better).
    for i, method in enumerate(methods):
        means = [summary[(summary.map_kind == m) & (summary.method == method)].redundant_ratio.mean()
                 for m in MAPS]
        ax1.bar(x + (i - 1) * width, means, width, label=method, color=METHOD_COLORS[method])
    ax1.set_xticks(x); ax1.set_xticklabels(MAPS)
    ax1.set_ylabel("redundant-coverage ratio")
    ax1.set_title("(a) Redundant coverage by allocation (lower = better)")
    ax1.legend(title="method"); ax1.grid(axis="y", alpha=0.3)

    # (b) time-to-90% coverage (lower = faster).
    for i, method in enumerate(methods):
        means = [_mean_ttc(summary[(summary.map_kind == m) & (summary.method == method)]) for m in MAPS]
        ax2.bar(x + (i - 1) * width, means, width, label=method, color=METHOD_COLORS[method])
    ax2.set_xticks(x); ax2.set_xticklabels(MAPS)
    ax2.set_ylabel("steps to 90% coverage (mean, uncensored)")
    ax2.set_title("(b) Time-to-90% by allocation (lower = faster)")
    ax2.legend(title="method"); ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("E2: allocation ablation (2 agents, 30 held-out seeds/map)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e2_allocation_ablation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# E3 — team size at matched flight time
# --------------------------------------------------------------------------- #
def plot_e3(steps: pd.DataFrame) -> None:
    """Coverage vs *total flight time* (step x n_agents), so a team only 'wins'
    if it explores more per unit of combined effort, not just per wall-clock step."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    for ax, map_kind in zip(axes.ravel(), MAPS):
        sub = steps[steps.map_kind == map_kind]
        for n in (1, 2, 3):
            nsub = sub[sub.n_agents == n]
            if not len(nsub):
                continue
            # x-axis = combined flight time = step * n_agents.
            nsub = nsub.assign(flight=nsub.step * n)
            max_flight = int(nsub.flight.max())
            grid = np.linspace(1, max_flight, 200)
            curves = []
            for _seed, g in nsub.groupby("seed"):
                g = g.sort_values("flight")
                curves.append(np.interp(grid, g.flight.values, g.coverage.values,
                                        left=0.0, right=g.coverage.values[-1]))
            ax.plot(grid, np.mean(curves, axis=0), color=N_COLORS[n], label=f"N={n}", linewidth=2)
        ax.axhline(90, color="grey", linestyle=":", linewidth=1)
        ax.set_title(map_kind); ax.set_xlabel("total flight time (step x N)")
        ax.set_ylabel("coverage (%)"); ax.grid(alpha=0.3)
    axes.ravel()[0].legend(title="team size", loc="lower right")
    fig.suptitle("E3: team size at matched flight time (hungarian, 30 seeds/map)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e3_team_size.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# E4 — lambda sensitivity
# --------------------------------------------------------------------------- #
def plot_e4(summary: pd.DataFrame) -> None:
    lambdas = sorted(summary.lambda_cost.unique())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for map_kind in MAPS:
        sub = summary[summary.map_kind == map_kind]
        red = [sub[sub.lambda_cost == lam].redundant_ratio.mean() for lam in lambdas]
        ttc = [_mean_ttc(sub[sub.lambda_cost == lam]) for lam in lambdas]
        ax1.plot(lambdas, red, marker="o", label=map_kind)
        ax2.plot(lambdas, ttc, marker="o", label=map_kind)
    ax1.set_xlabel("lambda"); ax1.set_ylabel("redundant-coverage ratio")
    ax1.set_title("(a) Redundant coverage vs lambda"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.set_xlabel("lambda"); ax2.set_ylabel("steps to 90% (mean, uncensored)")
    ax2.set_title("(b) Time-to-90% vs lambda"); ax2.legend(); ax2.grid(alpha=0.3)
    fig.suptitle("E4: lambda sensitivity (hungarian, 2 agents, 30 seeds/map)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e4_lambda_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# E5 — exploration strategy vs detection outcomes
# --------------------------------------------------------------------------- #
def plot_e5(summary: pd.DataFrame) -> None:
    methods = ["none", "greedy", "hungarian"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(MAPS))
    width = 0.25
    # (a) fraction of targets localised (higher = better).
    for i, method in enumerate(methods):
        means = [summary[(summary.map_kind == m) & (summary.method == method)].frac_localised.mean()
                 for m in MAPS]
        ax1.bar(x + (i - 1) * width, means, width, label=method, color=METHOD_COLORS[method])
    ax1.set_xticks(x); ax1.set_xticklabels(MAPS)
    ax1.set_ylabel("fraction of targets localised"); ax1.set_ylim(0, 1)
    ax1.set_title("(a) Targets localised by strategy (higher = better)")
    ax1.legend(title="method"); ax1.grid(axis="y", alpha=0.3)
    # (b) time-to-first-detection (lower = faster), uncensored mean.
    def _mean_ttfd(sub):
        v = pd.to_numeric(sub["time_to_first_detection"], errors="coerce").dropna()
        return float(v.mean()) if len(v) else float("nan")

    for i, method in enumerate(methods):
        means = [_mean_ttfd(summary[(summary.map_kind == m) & (summary.method == method)])
                 for m in MAPS]
        ax2.bar(x + (i - 1) * width, means, width, label=method, color=METHOD_COLORS[method])
    ax2.set_xticks(x); ax2.set_xticklabels(MAPS)
    ax2.set_ylabel("steps to first detection (mean, uncensored)")
    ax2.set_title("(b) Time-to-first-detection by strategy (lower = faster)")
    ax2.legend(title="method"); ax2.grid(axis="y", alpha=0.3)
    fig.suptitle("E5: exploration strategy vs detection (2 agents, 8 targets, 30 seeds/map)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e5_strategy_vs_detection.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    made = []
    if (RAW / "e2_summary.csv").exists():
        plot_e2(pd.read_csv(RAW / "e2_summary.csv")); made.append("e2_allocation_ablation.png")
    if (RAW / "e3_steps.csv").exists():
        plot_e3(pd.read_csv(RAW / "e3_steps.csv")); made.append("e3_team_size.png")
    if (RAW / "e4_summary.csv").exists():
        plot_e4(pd.read_csv(RAW / "e4_summary.csv")); made.append("e4_lambda_sweep.png")
    if (RAW / "e5_summary.csv").exists():
        plot_e5(pd.read_csv(RAW / "e5_summary.csv")); made.append("e5_strategy_vs_detection.png")
    print(f"wrote {len(made)} figures to {FIG_DIR}/: {', '.join(made) or '(no input CSVs found)'}")


if __name__ == "__main__":
    main()
