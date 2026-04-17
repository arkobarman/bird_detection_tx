"""
Generate per-orthomosaic species distribution bar charts.

Reads individual OM_xxx_annotations.json files and generates a distribution
chart for each orthomosaic showing species counts.

Usage:
    python scripts/visualization/plot_om_distributions.py

    # Custom paths
    python scripts/visualization/plot_om_distributions.py \
        --annotations-dir data/BirdDataset_2025_10k/annotations \
        --output-dir figures/data_exploration/om_distributions

Output:
    figures/data_exploration/om_distributions/
    ├── OM_001_distribution.png
    ├── OM_002_distribution.png
    └── ...
"""

import argparse
import json
from pathlib import Path
from collections import Counter

import pandas as pd
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


def plot_om_distribution(om_id: str, species_counts: Counter, output_path: Path):
    """Generate and save a distribution chart for a single orthomosaic."""
    if not species_counts:
        print(f"[warn] No annotations for {om_id}, skipping")
        return

    # Sort by count
    sorted_species = species_counts.most_common()
    species_codes = [s[0] for s in sorted_species]
    counts = [s[1] for s in sorted_species]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(4, len(species_codes) * 0.35)))

    # Color gradient
    colors = plt.cm.viridis(np.linspace(0.9, 0.2, len(species_codes)))

    # Bar chart
    bars = ax.barh(range(len(species_codes)), counts, color=colors)

    # Labels with full names
    labels = [f"{code} - {SPECIES_NAMES.get(code, code)}" for code in species_codes]
    ax.set_yticks(range(len(species_codes)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()

    # Add count labels
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
    ax.set_xlabel("Number of Annotations", fontsize=11)
    ax.set_title(f"{om_id}: Species Distribution\n(Total: {total:,} annotations, {len(species_codes)} species)",
                 fontsize=12, fontweight="bold")

    ax.set_xlim(0, max_count * 1.12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate per-orthomosaic species distribution charts")
    parser.add_argument("--annotations-dir", type=str,
                        default="data/BirdDataset_2025_10k/annotations",
                        help="Directory containing OM_xxx_annotations.json files")
    parser.add_argument("--output-dir", type=str,
                        default="figures/data_exploration/om_distributions",
                        help="Output directory for distribution charts")
    args = parser.parse_args()

    annotations_dir = Path(args.annotations_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all OM annotation files
    om_files = sorted(annotations_dir.glob("OM_*_annotations.json"))
    print(f"[info] Found {len(om_files)} orthomosaic annotation files in {annotations_dir}")

    if not om_files:
        print("[error] No OM_xxx_annotations.json files found")
        return

    # Process each orthomosaic
    summary_data = []

    for om_file in om_files:
        om_id = om_file.stem.replace("_annotations", "")  # e.g., "OM_001"

        # Load annotations
        with open(om_file, encoding="utf-8") as f:
            data = json.load(f)

        # Count species
        species_counts = Counter()
        for ann in data.get("annotations", []):
            species = ann.get("tcws_species", "UNKNOWN")
            species_counts[species] += 1

        # Generate chart
        output_path = output_dir / f"{om_id}_distribution.png"
        plot_om_distribution(om_id, species_counts, output_path)

        total = sum(species_counts.values())
        num_species = len(species_counts)
        summary_data.append({
            "orthomosaic": om_id,
            "total_annotations": total,
            "num_species": num_species,
            "top_species": species_counts.most_common(1)[0][0] if species_counts else "N/A",
        })

        print(f"[done] {om_id}: {total:,} annotations, {num_species} species -> {output_path.name}")

    # Save summary CSV
    summary_path = output_dir / "om_summary.csv"
    pd.DataFrame(summary_data).to_csv(summary_path, index=False)
    print(f"\n[done] Summary saved to {summary_path}")
    print(f"[done] Generated {len(om_files)} distribution charts in {output_dir}/")


if __name__ == "__main__":
    main()
