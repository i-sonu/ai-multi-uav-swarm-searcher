"""Plot publication-quality figures for Experiment E2 (Allocation Strategy Ablation)."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import REPO_ROOT

def main():
    raw_csv = REPO_ROOT / "results" / "raw" / "e2_allocation_ablation.csv"
    if not raw_csv.exists():
        print(f"Error: {raw_csv} not found. Run scripts/run_e2.py first.")
        return

    df = pd.read_csv(raw_csv)
    fig_dir = REPO_ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. Redundant Coverage Ratio Bar Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=df,
        x="map_kind",
        y="redundant_ratio",
        hue="allocation_method",
        palette="viridis",
        ax=ax,
    )
    ax.set_title("Experiment E2: Redundant Coverage Ratio by Allocation Strategy")
    ax.set_xlabel("Map Topology")
    ax.set_ylabel("Redundant Coverage Ratio (Lower is Better)")
    plt.tight_layout()
    fig_path1 = fig_dir / "e2_redundant_coverage.png"
    plt.savefig(fig_path1, dpi=300)
    plt.close()

    # 2. Time to 90% Coverage Box Plot
    df_valid = df.dropna(subset=["time_to_90"]).copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=df_valid,
        x="map_kind",
        y="time_to_90",
        hue="allocation_method",
        palette="magma",
        ax=ax,
    )
    ax.set_title("Experiment E2: Time-to-90% Coverage (Steps)")
    ax.set_xlabel("Map Topology")
    ax.set_ylabel("Steps to 90% Coverage (Lower is Better)")
    plt.tight_layout()
    fig_path2 = fig_dir / "e2_time_to_90.png"
    plt.savefig(fig_path2, dpi=300)
    plt.close()

    print(f"E2 figures generated successfully in {fig_dir}:")
    print(f"  - {fig_path1.name}")
    print(f"  - {fig_path2.name}")

if __name__ == "__main__":
    main()
