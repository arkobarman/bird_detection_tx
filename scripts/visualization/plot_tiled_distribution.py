"""
Generate species distribution bar chart for the TILED 500x500 dataset.

This script reads COCO-format annotations (all_annotations.json) from the tiled
dataset and counts species by category_id. Note that birds on tile boundaries
may appear in multiple tiles, so total counts will be higher than the original
orthomosaic dataset (~16% inflation).

Input:
    data/BirdDataset_2025_nonoverlapping_tiles_500/all_annotations.json
    - COCO format with categories, images, and annotations
    - Each annotation has category_id mapped to species

Output:
    figures/data_exploration/tiled_dataset_distribution.png (default)

Usage:
    python scripts/visualization/plot_tiled_distribution.py

    # Custom paths
    python scripts/visualization/plot_tiled_distribution.py \
        --json data/BirdDataset_2025_nonoverlapping_tiles_500/all_annotations.json \
        --output figures/data_exploration/tiled_dataset_distribution.png \
        --title "Custom Title"

See also:
    plot_species_distribution.py - For original 10k orthomosaic distribution (no duplicates)
"""

import argparse
import json
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


# Species code to full name mapping
SPECIES_NAMES = {
    "ROTEA": "Royal Tern Adults",
    "SATEA": "Sandwich Tern Adults",
    "BRPEC": "Brown Pelican Chicks",
    "LAGUA": "Laughing Gull Adults",
    "BRPEA": "Brown Pelican Adults",
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
    "REEGWMA": "Reddish Egret White Morph Adults",
    "REEGF": "Reddish Egret Flying",
    "REEGA": "Reddish Egret Adults",
    "NECOA": "Neotropic Cormorant Adults",
    "GREGF": "Great Egret Flying",
    "DCCOF": "Double-Crested Cormorant Flying",
    "BNSTA": "Black-Necked Stilt Adults",
}


def main():
    parser = argparse.ArgumentParser(description="Generate species distribution from COCO annotations")
    parser.add_argument("--json", type=str,
                        default="data/BirdDataset_2025_nonoverlapping_tiles_500/all_annotations.json",
                        help="Path to COCO-format all_annotations.json")
    parser.add_argument("--output", type=str,
                        default="figures/data_exploration/tiled_dataset_distribution.png",
                        help="Output path for the distribution chart")
    parser.add_argument("--title", type=str,
                        default="Tiled Dataset (500x500): Species Distribution",
                        help="Chart title")
    args = parser.parse_args()

    json_path = Path(args.json)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load COCO annotations
    print(f"[info] Loading annotations from {json_path}")
    with open(json_path, encoding="utf-8") as f:
        coco = json.load(f)

    # Build category lookup
    id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]}

    # Count species
    species_counts = Counter()
    for ann in coco["annotations"]:
        species = id_to_name.get(ann["category_id"], "UNKNOWN")
        species_counts[species] += 1

    print(f"[info] Total annotations: {sum(species_counts.values()):,}")
    print(f"[info] Number of species: {len(species_counts)}")

    # Sort by count
    sorted_species = species_counts.most_common()
    species_codes = [s[0] for s in sorted_species]
    counts = [s[1] for s in sorted_species]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Color gradient
    colors = plt.cm.viridis(np.linspace(0.9, 0.2, len(species_codes)))

    # Bar chart
    bars = ax.barh(range(len(species_codes)), counts, color=colors)

    # Labels with full names
    labels = [f"{code} - {SPECIES_NAMES.get(code, code)}" for code in species_codes]
    ax.set_yticks(range(len(species_codes)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()

    # Add count labels on bars
    max_count = max(counts)
    for i, (count, bar) in enumerate(zip(counts, bars)):
        if count > max_count * 0.15:
            ax.text(count - max_count * 0.02, i, f"{count:,}",
                    va="center", ha="right", fontsize=8, color="white", fontweight="bold")
        else:
            ax.text(count + max_count * 0.01, i, f"{count:,}",
                    va="center", ha="left", fontsize=8, color="black")

    # Formatting
    total = sum(counts)
    ax.set_xlabel("Number of Annotations", fontsize=12)
    ax.set_title(f"{args.title}\n(Total: {total:,} annotations across {len(species_codes)} species)",
                 fontsize=14, fontweight="bold")

    ax.set_xlim(0, max_count * 1.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"[done] Saved figure to {output_path}")

    # Print summary stats
    print("\n[info] Summary statistics:")
    print(f"  Most common: {species_codes[0]} ({counts[0]:,})")
    print(f"  Least common: {species_codes[-1]} ({counts[-1]:,})")
    median_count = sorted(counts)[len(counts) // 2]
    print(f"  Median count: {median_count}")
    print(f"  Species with <10 samples: {sum(1 for c in counts if c < 10)}")
    print(f"  Species with <50 samples: {sum(1 for c in counts if c < 50)}")


if __name__ == "__main__":
    main()
