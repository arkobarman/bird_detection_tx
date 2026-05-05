"""
Generate split distribution visualizations for a detection_tile_splits_* directory.

Input:
    <split_dir>/  (e.g. splits/detection_tile_splits_05042026/)
        train_annotations.json, val_annotations.json, test_annotations.json
        split_metadata.json

Output:
    --output-dir figures/data_exploration/MMDDYYYY/split_distribution/
        split_overview.png
        species_distribution_comparison.png
        orthomosaic_tile_assignment.png
        annotation_density.png

Usage:
    python scripts/visualization/plot_split_distribution.py \
        --split-dir splits/detection_tile_splits_05042026 \
        --output-dir figures/data_exploration/05042026/split_distribution
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


SPLIT_COLORS = {"train": "#2196F3", "val": "#FF9800", "test": "#4CAF50"}
SPLIT_LABELS = {"train": "Train", "val": "Val", "test": "Test"}

SPECIES_NAMES = {
    "ROTEA": "Royal Tern Adults",       "SATEA": "Sandwich Tern Adults",
    "BRPEC": "Brown Pelican Chicks",    "LAGUA": "Laughing Gull Adults",
    "BRPEA": "Brown Pelican Adults",    "BLSKA": "Black Skimmer Adults",
    "TRHEA": "Tri-Colored Heron Adults","GREGC": "Great Egret Chicks",
    "GREGA": "Great Egret Adults",      "LWBBA": "Large White Bird Above",
    "GBHEC": "Great Blue Heron Chicks", "RUTUA": "Ruddy Turnstone Adults",
    "WHIBA": "White Ibis Adults",       "ROSPA": "Roseate Spoonbill Adults",
    "OTHRA": "Other",                   "UNSURE": "Unsure",
    "GBHEA": "Great Blue Heron Adults", "DCCOA": "DC Cormorant Adults",
    "SNEGA": "Snowy Egret Adults",      "CATEA": "Caspian Tern Adults",
    "AMAVA": "American Avocet Adults",  "ROSPC": "Roseate Spoonbill Chicks",
    "REEGWM": "Reddish Egret W.Morph", "REEGA": "Reddish Egret Adults",
    "BLSTA": "Black-necked Stilt Adults",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_splits(split_dir: Path) -> dict:
    data = {}
    for s in ("train", "val", "test"):
        with open(split_dir / f"{s}_annotations.json") as f:
            data[s] = json.load(f)
    with open(split_dir / "split_metadata.json") as f:
        data["metadata"] = json.load(f)
    return data


def build_category_map(data: dict) -> dict:
    cats = data["train"]["categories"]
    return {c["id"]: c["name"] for c in cats}


def species_counts(split_data: dict, cat_map: dict) -> Counter:
    counts: Counter = Counter()
    for ann in split_data["annotations"]:
        counts[cat_map[ann["category_id"]]] += 1
    return counts


def tile_annotation_counts(split_data: dict) -> list[int]:
    img_counts: Counter = Counter()
    for ann in split_data["annotations"]:
        img_counts[ann["image_id"]] += 1
    all_counts = [img_counts.get(img["id"], 0) for img in split_data["images"]]
    return all_counts


# ── Plot 1: Overview ──────────────────────────────────────────────────────────

def plot_overview(data: dict, output_path: Path):
    meta = data["metadata"]
    sizes = meta["split_sizes"]
    score_details = meta["score_details"]
    missing = meta["missing_species"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Split Overview", fontsize=16, fontweight="bold", y=1.02)

    # ── Tile & annotation counts ──────────────────────────────────────────
    ax = axes[0]
    splits = ["train", "val", "test"]
    tile_counts  = [sizes["train_tiles"],       sizes["val_tiles"],       sizes["test_tiles"]]
    anno_counts  = [sizes["train_annotations"], sizes["val_annotations"], sizes["test_annotations"]]
    x = np.arange(3)
    w = 0.35
    bars1 = ax.bar(x - w/2, tile_counts, w, label="Tiles",       color=[SPLIT_COLORS[s] for s in splits], alpha=0.85)
    bars2 = ax.bar(x + w/2, anno_counts, w, label="Annotations", color=[SPLIT_COLORS[s] for s in splits], alpha=0.45, hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels([SPLIT_LABELS[s] for s in splits])
    ax.set_ylabel("Count")
    ax.set_title("Tiles & Annotations per Split")
    ax.legend(fontsize=9)
    for bar, count in zip(list(bars1) + list(bars2), tile_counts + anno_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{count:,}", ha="center", va="bottom", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Score breakdown ───────────────────────────────────────────────────
    ax = axes[1]
    score_names  = ["Distribution\nScore", "Missing Species\nPenalty", "Imbalance\nPenalty"]
    score_values = [
        score_details["distribution_score"],
        score_details["missing_penalty"],
        score_details["imbalance_penalty"],
    ]
    colors = ["#42A5F5", "#EF5350", "#FFA726"]
    bars = ax.bar(score_names, score_values, color=colors, alpha=0.85)
    ax.set_ylabel("Score (lower = better)")
    ax.set_title(f"Optimization Score Breakdown\nTotal: {meta['best_score']:.4f}  ({meta['iterations']:,} iterations)")
    for bar, val in zip(bars, score_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Missing species ───────────────────────────────────────────────────
    ax = axes[2]
    ax.axis("off")
    all_missing = set(missing.get("val", [])) | set(missing.get("test", []))
    if all_missing:
        lines = ["Missing species\n"]
        for sp in sorted(all_missing):
            in_val  = "✗ val"  if sp in missing.get("val",  []) else ""
            in_test = "✗ test" if sp in missing.get("test", []) else ""
            gaps = ", ".join(filter(None, [in_val, in_test]))
            lines.append(f"  {sp}  ({gaps})")
        ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
                fontsize=10, verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="#FFF3E0", alpha=0.8))
    else:
        ax.text(0.5, 0.5, "No missing species", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="green")
    ax.set_title("Coverage Gaps")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[done] {output_path.name}")


# ── Plot 2: Species distribution comparison ───────────────────────────────────

def plot_species_comparison(data: dict, cat_map: dict, output_path: Path):
    splits = ("train", "val", "test")
    counts = {s: species_counts(data[s], cat_map) for s in splits}
    all_species = sorted(
        set().union(*[c.keys() for c in counts.values()]),
        key=lambda sp: -(counts["train"][sp] + counts["val"][sp] + counts["test"][sp])
    )

    # Compute % of each split's total for each species
    totals = {s: sum(counts[s].values()) for s in splits}
    target = {"train": 0.80, "val": 0.10, "test": 0.10}

    n = len(all_species)
    fig, axes = plt.subplots(1, 2, figsize=(18, max(6, n * 0.38)))
    fig.suptitle("Species Distribution Across Splits", fontsize=14, fontweight="bold")

    # Left: absolute counts stacked bar
    ax = axes[0]
    y = np.arange(n)
    left = np.zeros(n)
    for s in splits:
        vals = np.array([counts[s].get(sp, 0) for sp in all_species], dtype=float)
        ax.barh(y, vals, left=left, color=SPLIT_COLORS[s], label=SPLIT_LABELS[s], alpha=0.85)
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels([f"{sp}  {SPECIES_NAMES.get(sp, '')}" for sp in all_species], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Annotation count")
    ax.set_title("Absolute Counts")
    ax.legend(loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right: % deviation from target (80/10/10)
    ax = axes[1]
    bar_h = 0.25
    for i, s in enumerate(splits):
        deviations = []
        for sp in all_species:
            total_sp = sum(counts[ss].get(sp, 0) for ss in splits)
            if total_sp == 0:
                deviations.append(0.0)
            else:
                actual_pct = counts[s].get(sp, 0) / total_sp * 100
                deviations.append(actual_pct - target[s] * 100)
        offset = (i - 1) * bar_h
        ax.barh(y + offset, deviations, bar_h, color=SPLIT_COLORS[s],
                label=f"{SPLIT_LABELS[s]} (target {target[s]*100:.0f}%)", alpha=0.85)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(all_species, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("% deviation from target")
    ax.set_title("% Deviation from Target (80/10/10)")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Shade missing species rows
    meta = data["metadata"]
    missing_all = set(meta["missing_species"].get("val", [])) | set(meta["missing_species"].get("test", []))
    for i, sp in enumerate(all_species):
        if sp in missing_all:
            for ax_ in axes:
                ax_.axhspan(i - 0.5, i + 0.5, color="#FFCDD2", alpha=0.35, zorder=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[done] {output_path.name}")


# ── Plot 3: Orthomosaic tile assignment grid ──────────────────────────────────

def plot_tile_assignment(data: dict, output_path: Path):
    # Build (om_id, tile_x, tile_y) → split mapping
    tile_split: dict[tuple, str] = {}
    for s in ("train", "val", "test"):
        for img in data[s]["images"]:
            key = (img["orthomosaic_id"], img["tile_x"], img["tile_y"])
            tile_split[key] = s

    om_ids = sorted(set(k[0] for k in tile_split))
    n_oms = len(om_ids)
    cols = 6
    rows = (n_oms + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.8, rows * 2.8))
    fig.suptitle("Tile Assignment per Orthomosaic  (blue=train, orange=val, green=test)",
                 fontsize=13, fontweight="bold")
    axes_flat = axes.flatten() if n_oms > 1 else [axes]

    color_map = {"train": np.array([33, 150, 243]) / 255,
                 "val":   np.array([255, 152,   0]) / 255,
                 "test":  np.array([76, 175,  80]) / 255}

    for idx, om_id in enumerate(om_ids):
        ax = axes_flat[idx]
        tiles = {(k[1], k[2]): v for k, v in tile_split.items() if k[0] == om_id}
        if not tiles:
            ax.axis("off")
            continue
        max_x = max(k[0] for k in tiles) + 1
        max_y = max(k[1] for k in tiles) + 1
        grid = np.ones((max_y, max_x, 3)) * 0.92  # light grey default
        for (tx, ty), s in tiles.items():
            grid[ty, tx] = color_map[s]
        ax.imshow(grid, aspect="equal", interpolation="nearest")
        ax.set_title(om_id, fontsize=9, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(n_oms, len(axes_flat)):
        axes_flat[idx].axis("off")

    legend_patches = [mpatches.Patch(color=SPLIT_COLORS[s], label=SPLIT_LABELS[s])
                      for s in ("train", "val", "test")]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[done] {output_path.name}")


# ── Plot 4: Annotation density distribution ───────────────────────────────────

def plot_annotation_density(data: dict, output_path: Path):
    splits = ("train", "val", "test")
    all_counts = {s: tile_annotation_counts(data[s]) for s in splits}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Annotation Density per Tile Across Splits", fontsize=14, fontweight="bold")

    # Left: violin plot of per-tile annotation counts
    ax = axes[0]
    parts = ax.violinplot([all_counts[s] for s in splits], positions=range(3),
                          showmedians=True, showextrema=True)
    for i, (pc, s) in enumerate(zip(parts["bodies"], splits)):
        pc.set_facecolor(SPLIT_COLORS[s])
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2)
    ax.set_xticks(range(3))
    ax.set_xticklabels([SPLIT_LABELS[s] for s in splits])
    ax.set_ylabel("Annotations per tile")
    ax.set_title("Distribution (all tiles including empty)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right: histogram of non-empty tiles only
    ax = axes[1]
    for s in splits:
        non_empty = [c for c in all_counts[s] if c > 0]
        ax.hist(non_empty, bins=30, color=SPLIT_COLORS[s], alpha=0.55,
                label=f"{SPLIT_LABELS[s]} (n={len(non_empty):,})", density=True)
    ax.set_xlabel("Annotations per tile (non-empty only)")
    ax.set_ylabel("Density")
    ax.set_title("Annotation Count Distribution (non-empty tiles)")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Summary table below
    for ax_ in axes:
        stats_text = []
        for s in splits:
            c = all_counts[s]
            non_empty = [x for x in c if x > 0]
            pct_empty = (len(c) - len(non_empty)) / len(c) * 100
            stats_text.append(
                f"{SPLIT_LABELS[s]}: mean={np.mean(c):.1f}  "
                f"median={int(np.median(c))}  "
                f"empty={pct_empty:.0f}%"
            )

    fig.text(0.5, -0.04, "\n".join(stats_text), ha="center", fontsize=9,
             fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#F5F5F5", alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[done] {output_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    repo_root = Path(__file__).parent.parent.parent

    parser = argparse.ArgumentParser(description="Visualize train/val/test split distribution")
    parser.add_argument("--split-dir",  type=str, required=True,
                        help="Path to detection_tile_splits_* directory")
    parser.add_argument("--output-dir", type=str,
                        default=str(repo_root / "figures/data_exploration/split_distribution"),
                        help="Output directory for figures")
    args = parser.parse_args()

    split_dir  = Path(args.split_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading splits from {split_dir}...")
    data    = load_splits(split_dir)
    cat_map = build_category_map(data)
    print(f"  train: {len(data['train']['images']):,} tiles, {len(data['train']['annotations']):,} annotations")
    print(f"  val:   {len(data['val']['images']):,} tiles, {len(data['val']['annotations']):,} annotations")
    print(f"  test:  {len(data['test']['images']):,} tiles, {len(data['test']['annotations']):,} annotations")
    print()

    plot_overview(data, output_dir / "split_overview.png")
    plot_species_comparison(data, cat_map, output_dir / "species_distribution_comparison.png")
    plot_tile_assignment(data, output_dir / "orthomosaic_tile_assignment.png")
    plot_annotation_density(data, output_dir / "annotation_density.png")

    print(f"\n[done] All figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
