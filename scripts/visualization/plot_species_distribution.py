"""
Generate species distribution bar chart.

Supports three input formats:
    --annotations-json  all_annotations.json from orthomosaic dataset (tcws_species detections)
    --crops-json        species_distribution.json from crop_coco_tiles.py output
    --csv-path          species_totals.csv with columns: Species, Count, Name

Output:
    --output   PATH     Output PNG path
    --paper            Publication-quality styling (300 dpi, larger fonts, log-scale x-axis,
                       threshold annotation at --threshold N)
    --threshold N      Draw a vertical threshold line at N samples (default: 50)
    --title    TEXT    Override chart title

Usage:
    # Crop dataset — paper figure
    python scripts/visualization/plot_species_distribution.py \
        --crops-json data/BirdDataset_2025_crops_500_05042026/species_distribution.json \
        --output figures/data_exploration/05042026/crop_species_distribution.png \
        --paper

    # Orthomosaic dataset
    python scripts/visualization/plot_species_distribution.py \
        --annotations-json data/BirdDataset_2025_10k/annotations/all_annotations.json \
        --output figures/data_exploration/05042026/original_dataset_distribution.png
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


SPECIES_NAMES = {
    "ROTEA": "Royal Tern",
    "SATEA": "Sandwich Tern",
    "BRPEC": "Brown Pelican (Chick)",
    "LAGUA": "Laughing Gull",
    "BRPEA": "Brown Pelican",
    "BLSKA": "Black Skimmer",
    "TRHEA": "Tricolored Heron",
    "GREGC": "Great Egret (Chick)",
    "GREGA": "Great Egret",
    "LWBBA": "Large White Bird (Above)",
    "MTRNS": "Mixed Terns",
    "GBHEC": "Great Blue Heron (Chick)",
    "RUTUA": "Ruddy Turnstone",
    "WHIBA": "White Ibis",
    "ROTEF": "Royal Tern (Flying)",
    "ROSPA": "Roseate Spoonbill",
    "OTHRA": "Other",
    "UNSURE": "Unidentified",
    "GBHEA": "Great Blue Heron",
    "DCCOA": "Double-crested Cormorant",
    "SATEF": "Sandwich Tern (Flying)",
    "LAGUF": "Laughing Gull (Flying)",
    "BRPEF": "Brown Pelican (Flying)",
    "SNEGA": "Snowy Egret",
    "CATEA": "Caspian Tern",
    "AMAVA": "American Avocet",
    "ROSPC": "Roseate Spoonbill (Chick)",
    "WHIBF": "White Ibis (Flying)",
    "RUTUF": "Ruddy Turnstone (Flying)",
    "REEGWM": "Reddish Egret (White Morph)",
    "REEGA": "Reddish Egret",
    "BLSTA": "Black-necked Stilt",
    "NECOA": "Neotropic Cormorant",
    "GREGF": "Great Egret (Flying)",
    "DCCOF": "Double-crested Cormorant (Flying)",
    "BNSTA": "Black-necked Stilt",
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


def load_from_crops_json(json_path: Path) -> pd.DataFrame:
    with open(json_path) as f:
        data = json.load(f)
    rows = [
        {"Species": sp, "Count": cnt, "Name": SPECIES_NAMES.get(sp, sp)}
        for sp, cnt in sorted(data["species_counts"].items(), key=lambda x: -x[1])
    ]
    return pd.DataFrame(rows)


def make_figure(df: pd.DataFrame, output_path: Path, title: str,
                paper: bool, threshold: int):

    n = len(df)
    fig_h = max(5, n * (0.45 if paper else 0.38))
    fig_w = 11 if paper else 14
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    if paper:
        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        })

    # Color: green for above threshold, muted grey-blue for below
    colors = [
        "#2E7D32" if row["Count"] >= threshold else "#B0BEC5"
        for _, row in df.iterrows()
    ]

    bars = ax.barh(range(n), df["Count"], color=colors,
                   edgecolor="white", linewidth=0.4, height=0.7)

    # Y-axis labels: "CODE  Full Name"
    ax.set_yticks(range(n))
    ax.set_yticklabels(
        [f"{row['Species']}   {row['Name']}" for _, row in df.iterrows()],
        fontsize=9 if paper else 9,
        fontfamily="monospace" if paper else None,
    )
    ax.invert_yaxis()

    # Count labels
    max_count = df["Count"].max()
    for i, (count, bar) in enumerate(zip(df["Count"], bars)):
        if paper:
            # Always place outside for clean look
            ax.text(count + max_count * 0.012, i, f"{count:,}",
                    va="center", ha="left", fontsize=8,
                    color="#333333" if count >= threshold else "#888888")
        else:
            if count > max_count * 0.15:
                ax.text(count - max_count * 0.02, i, f"{count:,}",
                        va="center", ha="right", fontsize=8,
                        color="white", fontweight="bold")
            else:
                ax.text(count + max_count * 0.01, i, f"{count:,}",
                        va="center", ha="left", fontsize=8, color="black")

    # Threshold line
    if threshold > 0:
        ax.axvline(threshold, color="#C62828", linewidth=1.2,
                   linestyle="--", zorder=3)
        ax.text(threshold + max_count * 0.008, -0.7,
                f"  ≥{threshold} threshold",
                color="#C62828", fontsize=8.5, va="top", ha="left")

    # Legend patches
    import matplotlib.patches as mpatches
    above = mpatches.Patch(color="#2E7D32", label=f"Included  (≥{threshold} crops)")
    below = mpatches.Patch(color="#B0BEC5", label=f"Excluded / merged  (<{threshold} crops)")
    ax.legend(handles=[above, below], fontsize=9,
              loc="lower right", framealpha=0.9)

    ax.set_xlabel("Number of annotated instances" if paper else "Number of Annotations",
                  fontsize=11 if paper else 12)
    ax.set_title(title, fontsize=12 if paper else 14,
                 fontweight="bold", pad=10)
    ax.set_xlim(0, max_count * 1.18)

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.xaxis.grid(True, linestyle="--", alpha=0.25 if paper else 0.3, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if paper:
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False)

    plt.tight_layout()
    dpi = 300 if paper else 150
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()


def main():
    repo_root = Path(__file__).parent.parent.parent

    parser = argparse.ArgumentParser(description="Generate species distribution chart")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--annotations-json", type=str,
                     default=str(repo_root / "data/BirdDataset_2025_10k/annotations/all_annotations.json"),
                     help="Path to all_annotations.json from the orthomosaic dataset")
    src.add_argument("--crops-json", type=str,
                     help="Path to species_distribution.json from crop_coco_tiles.py output")
    src.add_argument("--csv-path", type=str,
                     help="Path to species_totals.csv (columns: Species, Count, Name)")
    parser.add_argument("--output", type=str,
                        default=str(repo_root / "figures/data_exploration/species_distribution.png"),
                        help="Output PNG path")
    parser.add_argument("--title", type=str, default=None,
                        help="Override chart title")
    parser.add_argument("--paper", action="store_true",
                        help="Publication-quality styling (300 dpi, serif font, clean layout)")
    parser.add_argument("--threshold", type=int, default=50,
                        help="Draw inclusion threshold line at N samples (default: 50, 0 to hide)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.crops_json:
        df = load_from_crops_json(Path(args.crops_json))
        source = args.crops_json
        default_title = (
            "Chester Island 2025 — Cropped Instance Distribution\n"
            f"({df['Count'].sum():,} crops across {len(df)} species)"
        )
    elif args.csv_path:
        df = pd.read_csv(args.csv_path)
        source = args.csv_path
        default_title = (
            "Chester Island 2025 — Species Distribution\n"
            f"({df['Count'].sum():,} annotations across {len(df)} species)"
        )
    else:
        df = load_from_annotations_json(Path(args.annotations_json))
        source = args.annotations_json
        default_title = (
            "Chester Island 2025 — Orthomosaic Species Distribution\n"
            f"({df['Count'].sum():,} annotations across {len(df)} species)"
        )

    df = df.sort_values("Count", ascending=False).reset_index(drop=True)
    title = args.title if args.title else default_title

    print(f"[info] Loaded {len(df)} species from {source}")
    print(f"[info] Total: {df['Count'].sum():,}  |  threshold: {args.threshold}  |  paper: {args.paper}")

    make_figure(df, output_path, title, paper=args.paper, threshold=args.threshold)
    print(f"[done] Saved {'paper-quality ' if args.paper else ''}figure to {output_path}")

    print(f"\n[info] Above threshold (≥{args.threshold}): {(df['Count'] >= args.threshold).sum()} species")
    print(f"[info] Below threshold (<{args.threshold}): {(df['Count'] < args.threshold).sum()} species")
    print(f"[info] Median count: {df['Count'].median():.0f}")


if __name__ == "__main__":
    main()
