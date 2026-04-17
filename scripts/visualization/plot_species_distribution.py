"""
Generate species distribution bar chart for the Chester Island 2025 dataset.

Usage:
    python scripts/visualization/plot_species_distribution.py

Output:
    figures/data_exploration/original_dataset_distribution.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def main():
    # Paths
    repo_root = Path(__file__).parent.parent.parent
    csv_path = repo_root / "data/BirdDataset_2025_10k/metadata/species_totals.csv"
    output_dir = repo_root / "figures/data_exploration"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "original_dataset_distribution.png"

    # Load data
    df = pd.read_csv(csv_path)
    print(f"[info] Loaded {len(df)} species from {csv_path}")
    print(f"[info] Total annotations: {df['Count'].sum():,}")

    # Sort by count (already sorted but ensure)
    df = df.sort_values("Count", ascending=False).reset_index(drop=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Color gradient based on count (log scale for better visualization)
    colors = plt.cm.viridis(np.linspace(0.9, 0.2, len(df)))

    # Bar chart
    bars = ax.barh(range(len(df)), df["Count"], color=colors)

    # Labels
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f"{row['Species']} - {row['Name']}" for _, row in df.iterrows()], fontsize=9)
    ax.invert_yaxis()  # Largest at top

    # Add count labels on bars
    for i, (count, bar) in enumerate(zip(df["Count"], bars)):
        # Position label inside or outside bar depending on bar length
        if count > df["Count"].max() * 0.15:
            ax.text(count - df["Count"].max() * 0.02, i, f"{count:,}",
                    va="center", ha="right", fontsize=8, color="white", fontweight="bold")
        else:
            ax.text(count + df["Count"].max() * 0.01, i, f"{count:,}",
                    va="center", ha="left", fontsize=8, color="black")

    # Formatting
    ax.set_xlabel("Number of Annotations", fontsize=12)
    ax.set_title("Chester Island 2025 Dataset: Species Distribution\n(Total: {:,} annotations across {} species)".format(
        df["Count"].sum(), len(df)), fontsize=14, fontweight="bold")

    ax.set_xlim(0, df["Count"].max() * 1.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add grid
    ax.xaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"[done] Saved figure to {output_path}")

    # Also print summary stats
    print("\n[info] Summary statistics:")
    print(f"  Most common: {df.iloc[0]['Species']} ({df.iloc[0]['Count']:,})")
    print(f"  Least common: {df.iloc[-1]['Species']} ({df.iloc[-1]['Count']:,})")
    print(f"  Median count: {df['Count'].median():.0f}")
    print(f"  Species with <10 samples: {(df['Count'] < 10).sum()}")
    print(f"  Species with <50 samples: {(df['Count'] < 50).sum()}")


if __name__ == "__main__":
    main()
