"""
Generate species distribution bar chart for the ORIGINAL 10k orthomosaic dataset.

Reads either a pre-computed species_totals.csv OR directly from all_annotations.json.
Each bird is counted exactly once (no duplicates from tiling).

Input (one of):
    --annotations-json  data/BirdDataset_2025_10k/annotations/all_annotations.json
    --csv-path          data/BirdDataset_2025_10k/metadata/species_totals.csv

Output:
    --output  figures/data_exploration/original_dataset_distribution.png

Usage:
    python scripts/visualization/plot_species_distribution.py \
        --annotations-json data/BirdDataset_2025_10k/annotations/all_annotations.json \
        --output figures/data_exploration/05042026/original_dataset_distribution.png

See also:
    plot_tiled_distribution.py - For tiled (500x500) dataset distribution
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


SPECIES_NAMES = {
    "ROTEA": "Royal Tern Adults",
    "SATEA": "Sandwich Tern Adults",
    "BRPEC": "Brown Pelican Chicks",
    "LAGUA": "Laughing Gull Adults",
    "BRPEA": "Brown Pelican Adults",
    "BLSKA": "Black Skimmer Adults",
    "TRHEA": "Tri-Colored Heron Adults",
    "GREGC": "Great Egret Chicks",
    "GREGA": "Great Egret Adults",
    "LWBBA": "Large White Bird Above Canopy",
    "MTRNS": "Mixed Terns (Nesting/Sitting)",
    "GBHEC": "Great Blue Heron Chicks",
    "RUTUA": "Ruddy Turnstone Adults",
    "WHIBA": "White Ibis Adults",
    "ROTEF": "Royal Tern Flying",
    "ROSPA": "Roseate Spoonbill Adults",
    "OTHRA": "Other (Not nesting species)",
    "UNSURE": "Unsure/Unidentified",
    "GBHEA": "Great Blue Heron Adults",
    "DCCOA": "Double-Crested Cormorant Adults",
    "SATEF": "Sandwich Tern Flying",
    "LAGUF": "Laughing Gull Flying",
    "BRPEF": "Brown Pelican Flying",
    "SNEGA": "Snowy Egret Adults",
    "CATEA": "Caspian Tern Adults",
    "AMAVA": "American Avocet Adults",
    "ROSPC": "Roseate Spoonbill Chicks",
    "WHIBF": "White Ibis Flying",
    "RUTUF": "Ruddy Turnstone Flying",
    "REEGWM": "Reddish Egret White Morph",
    "REEGA": "Reddish Egret Adults",
    "BLSTA": "Black-necked Stilt Adults",
    "NECOA": "Neotropic Cormorant Adults",
    "GREGF": "Great Egret Flying",
    "DCCOF": "Double-Crested Cormorant Flying",
    "BNSTA": "Black-necked Stilt Adults",
}


def load_from_annotations_json(json_path: Path) -> pd.DataFrame:
    with open(json_path) as f:
        data = json.load(f)
    counts: Counter = Counter()
    for img in data["images"]:
        for det in img.get("detections", []):
            counts[det.get("tcws_species", "UNKN")] += 1
    rows = [
        {"Species": sp, "Count": cnt, "Name": SPECIES_NAMES.get(sp, sp)}
        for sp, cnt in counts.most_common()
    ]
    return pd.DataFrame(rows)


def main():
    repo_root = Path(__file__).parent.parent.parent

    parser = argparse.ArgumentParser(description="Generate orthomosaic species distribution chart")
    parser.add_argument("--annotations-json", type=str,
                        default=str(repo_root / "data/BirdDataset_2025_10k/annotations/all_annotations.json"),
                        help="Path to all_annotations.json from the orthomosaic dataset")
    parser.add_argument("--csv-path", type=str, default=None,
                        help="Path to species_totals.csv (overrides --annotations-json if provided)")
    parser.add_argument("--output", type=str,
                        default=str(repo_root / "figures/data_exploration/original_dataset_distribution.png"),
                        help="Output PNG path")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    if args.csv_path:
        df = pd.read_csv(args.csv_path)
        print(f"[info] Loaded {len(df)} species from {args.csv_path}")
    else:
        df = load_from_annotations_json(Path(args.annotations_json))
        print(f"[info] Loaded {len(df)} species from {args.annotations_json}")
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
