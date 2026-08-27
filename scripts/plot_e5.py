"""Plot publication-quality figures for Experiment E5 (Target Detection & Localization)."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import REPO_ROOT


def main():
    raw_csv = REPO_ROOT / "results" / "raw" / "e5_target_registration.csv"
    if not raw_csv.exists():
        print(f"Error: {raw_csv} not found. Run scripts/run_e5.py first.")
        return

    df = pd.read_csv(raw_csv)
    fig_dir = REPO_ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. Target Recall Bar Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=df,
        x="map_kind",
        y="target_recall",
        hue="allocation_method",
        palette="viridis",
        ax=ax,
    )
    ax.set_title("Experiment E5: Target Recall Rate by Allocation Strategy")
    ax.set_xlabel("Map Topology")
    ax.set_ylabel("Target Recall (%) (Higher is Better)")
    plt.tight_layout()
    fig_path1 = fig_dir / "e5_target_recall.png"
    plt.savefig(fig_path1, dpi=300)
    plt.close()

    # 2. Mean Localisation Error Bar Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=df,
        x="map_kind",
        y="loc_error_mean",
        hue="allocation_method",
        palette="crest",
        ax=ax,
    )
    ax.set_title("Experiment E5: Mean Spatial Localisation Error")
    ax.set_xlabel("Map Topology")
    ax.set_ylabel("Mean Position Error (m) (Lower is Better)")
    plt.tight_layout()
    fig_path2 = fig_dir / "e5_localisation_error.png"
    plt.savefig(fig_path2, dpi=300)
    plt.close()

    # 3. Time to First Detection Bar Plot
    df_valid = df.dropna(subset=["time_to_first_detect"]).copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=df_valid,
        x="map_kind",
        y="time_to_first_detect",
        hue="allocation_method",
        palette="rocket",
        ax=ax,
    )
    ax.set_title("Experiment E5: Steps to First Target Sighting")
    ax.set_xlabel("Map Topology")
    ax.set_ylabel("First Sighting Step (Lower is Better)")
    plt.tight_layout()
    fig_path3 = fig_dir / "e5_time_to_first_detect.png"
    plt.savefig(fig_path3, dpi=300)
    plt.close()

    print(f"E5 figures generated successfully in {fig_dir}:")
    print(f"  - {fig_path1.name}")
    print(f"  - {fig_path2.name}")
    print(f"  - {fig_path3.name}")


if __name__ == "__main__":
    main()
