"""Plot publication-quality figures for Experiment E4 (Lambda Sensitivity Sweep)."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import REPO_ROOT

def main():
    raw_csv = REPO_ROOT / "results" / "raw" / "e4_lambda_sweep.csv"
    if not raw_csv.exists():
        print(f"Error: {raw_csv} not found. Run scripts/run_e4.py first.")
        return

    df = pd.read_csv(raw_csv)
    fig_dir = REPO_ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", font_scale=1.1)

    # Path Length vs Lambda Trade-off
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=df,
        x="lambda_val",
        y="path_length_m",
        hue="map_kind",
        marker="s",
        linewidth=2.5,
        ax=ax,
    )
    ax.set_title("Experiment E4: Total Path Length (m) vs Lambda Weight")
    ax.set_xlabel("Lambda Cost Weight")
    ax.set_ylabel("Total Path Length Traveled (m)")
    plt.tight_layout()
    fig_path = fig_dir / "e4_lambda_path_length.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"E4 figure generated successfully: {fig_path}")

if __name__ == "__main__":
    main()
