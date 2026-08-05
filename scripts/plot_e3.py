"""Plot publication-quality figures for Experiment E3 (Team Size Scaling)."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import REPO_ROOT

def main():
    raw_csv = REPO_ROOT / "results" / "raw" / "e3_team_scaling.csv"
    if not raw_csv.exists():
        print(f"Error: {raw_csv} not found. Run scripts/run_e3.py first.")
        return

    df = pd.read_csv(raw_csv)
    fig_dir = REPO_ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", font_scale=1.1)

    # Time to 90% Coverage Speedup by Team Size
    df_valid = df.dropna(subset=["time_to_90"]).copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=df_valid,
        x="n_agents",
        y="time_to_90",
        hue="map_kind",
        marker="o",
        linewidth=2.5,
        ax=ax,
    )
    ax.set_title("Experiment E3: Exploration Speedup by Swarm Size N")
    ax.set_xlabel("Number of Agents (N)")
    ax.set_ylabel("Steps to 90% Coverage (Lower is Better)")
    ax.set_xticks([1, 2, 3])
    plt.tight_layout()
    fig_path = fig_dir / "e3_team_scaling_speedup.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"E3 figure generated successfully: {fig_path}")

if __name__ == "__main__":
    main()
