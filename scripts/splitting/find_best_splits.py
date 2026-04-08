#!/usr/bin/env python3
"""
Find Best Dataset Splits for Bird Detection Dataset

Generates train/val/test splits (80/10/10) at the tile level,
optimizing for species distribution similarity across splits.

Methodology:
    We use random search to find an optimal tile-level split. The algorithm:

    1. RANDOM SAMPLING: Generate N candidate splits by randomly shuffling all
       tiles and partitioning into 80% train / 10% val / 10% test.

    2. SCORING: Each candidate split is scored (lower = better) using:

       score = distribution_score + missing_penalty + imbalance_penalty

       Where:

       a) Distribution Score (L1 distance):
          For each split, compute the species proportion vector P_split.
          Compare to global distribution P_global using L1 distance:

            D_split = sum(|P_split[i] - P_global[i]|) for all species i
            distribution_score = D_train + D_val + D_test

          This measures how well each split preserves the overall species
          distribution. A perfect split would have D = 0 for all splits.

       b) Missing Species Penalty:
          For each species present globally but missing in val or test:

            penalty += sqrt(species_frequency) * weight

          The sqrt scaling ensures rare species aren't over-penalized while
          still encouraging their presence in evaluation sets. Weight = 2.0.

       c) Annotation Imbalance Penalty:
          Penalizes deviation from expected 80/10/10 annotation counts:

            imbalance = |train_ann - 0.8*total| + |val_ann - 0.1*total| + ...
            penalty = imbalance / total * weight

          Weight = 0.5. This is a soft constraint since tile-level splits
          may not perfectly align with annotation-level ratios.

    3. SELECTION: Track the split with the lowest score across all iterations.

    Why Random Search?
    - The search space is combinatorially large (C(8800, 7040) ways to choose train)
    - Stratified sampling is complex with 35 species and tile-level constraints
    - Random search with enough iterations reliably finds good solutions
    - Reproducible via fixed random seed

Usage:
    python find_best_splits.py \\
        --tiles_dir data/BirdDataset_2025_nonoverlapping_tiles_500 \\
        --annotations data/BirdDataset_2025_nonoverlapping_tiles_500/all_annotations.json \\
        --iterations 5000

Arguments:
    --tiles_dir, -t     Directory containing the tiled images
    --annotations, -a   Path to COCO-style annotations file (all_annotations.json)
    --iterations, -n    Number of random splits to try (default: 3000)
    --seed, -s          Random seed for reproducibility (default: 42)
    --output, -o        Output directory for splits (default: <tiles_dir>/splits)

Outputs:
    <output>/train_annotations.json   COCO annotations for training set
    <output>/val_annotations.json     COCO annotations for validation set
    <output>/test_annotations.json    COCO annotations for test set
    <output>/train_tiles.txt          List of training tile paths
    <output>/val_tiles.txt            List of validation tile paths
    <output>/test_tiles.txt           List of test tile paths
    <output>/split_report.txt         Detailed diagnostic report
    <output>/split_metadata.json      Reproducibility metadata
"""

import argparse
import json
import random
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np


def load_annotations(path: str) -> Dict[str, Any]:
    """Load COCO-style annotations file."""
    with open(path, "r") as f:
        return json.load(f)


def print_dataset_summary(
    images: List[Dict],
    annotations: List[Dict],
    categories: List[Dict]
) -> None:
    """Print a summary of the dataset."""
    # Count tiles per orthomosaic
    ortho_counts = Counter(img.get("orthomosaic_id", "unknown") for img in images)
    num_orthomosaics = len(ortho_counts)

    # Count tiles with/without annotations
    image_ids_with_annotations = set(ann["image_id"] for ann in annotations)
    tiles_with_birds = sum(1 for img in images if img["id"] in image_ids_with_annotations)
    tiles_without_birds = len(images) - tiles_with_birds

    # Calculate percentages
    pct_empty = 100 * tiles_without_birds / len(images)
    pct_with_birds = 100 * tiles_with_birds / len(images)

    # Calculate tiles per orthomosaic (assuming uniform)
    tiles_per_ortho = len(images) // num_orthomosaics if num_orthomosaics > 0 else 0

    print()
    print("Dataset Summary:")
    print(f"  - {len(images):,} tiles across {num_orthomosaics} orthomosaics ({tiles_per_ortho} tiles each)")
    print(f"  - {len(annotations):,} annotations across {len(categories)} species")
    print(f"  - {tiles_without_birds:,} tiles ({pct_empty:.0f}%) have zero annotations")
    print(f"  - {tiles_with_birds:,} tiles ({pct_with_birds:.0f}%) contain birds")
    print()


def compute_tile_species_counts(
    annotations: List[Dict],
    categories: List[Dict]
) -> Tuple[Dict[int, Dict[int, int]], Dict[int, str], List[int]]:
    """
    Compute per-tile species counts.

    Returns:
        tile_species: {image_id: {category_id: count}}
        cat_id_to_name: {category_id: name}
        all_cat_ids: sorted list of all category IDs
    """
    cat_id_to_name = {c["id"]: c["name"] for c in categories}
    all_cat_ids = sorted(cat_id_to_name.keys())

    # Build per-tile annotation counts
    tile_species = defaultdict(lambda: defaultdict(int))
    for ann in annotations:
        tile_species[ann["image_id"]][ann["category_id"]] += 1

    # Convert to regular dict
    tile_species = {k: dict(v) for k, v in tile_species.items()}

    return tile_species, cat_id_to_name, all_cat_ids


def compute_global_distribution(
    annotations: List[Dict],
    all_cat_ids: List[int]
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Compute global species distribution.

    Returns:
        proportions: numpy array of proportions for each category
        counts: {category_id: count}
    """
    counts = Counter(ann["category_id"] for ann in annotations)
    total = sum(counts.values())

    # Create proportion vector (same order as all_cat_ids)
    proportions = np.array([counts.get(cat_id, 0) / total for cat_id in all_cat_ids])

    return proportions, dict(counts)


def compute_split_distribution(
    image_ids: List[int],
    tile_species: Dict[int, Dict[int, int]],
    all_cat_ids: List[int]
) -> Tuple[np.ndarray, Dict[int, int], int]:
    """
    Compute species distribution for a set of tiles.

    Returns:
        proportions: numpy array of proportions
        counts: {category_id: count}
        total: total annotations
    """
    counts = Counter()
    for img_id in image_ids:
        if img_id in tile_species:
            for cat_id, count in tile_species[img_id].items():
                counts[cat_id] += count

    total = sum(counts.values())
    if total == 0:
        proportions = np.zeros(len(all_cat_ids))
    else:
        proportions = np.array([counts.get(cat_id, 0) / total for cat_id in all_cat_ids])

    return proportions, dict(counts), total


def score_split(
    train_ids: List[int],
    val_ids: List[int],
    test_ids: List[int],
    tile_species: Dict[int, Dict[int, int]],
    all_cat_ids: List[int],
    global_proportions: np.ndarray,
    global_counts: Dict[int, int],
    total_annotations: int,
    missing_penalty_weight: float = 2.0,
    imbalance_penalty_weight: float = 0.5
) -> Tuple[float, Dict[str, Any]]:
    """
    Score a candidate split.

    Lower score = better split.

    Returns:
        score: total score (lower is better)
        details: dict with breakdown
    """
    # Compute distributions for each split
    p_train, counts_train, total_train = compute_split_distribution(
        train_ids, tile_species, all_cat_ids
    )
    p_val, counts_val, total_val = compute_split_distribution(
        val_ids, tile_species, all_cat_ids
    )
    p_test, counts_test, total_test = compute_split_distribution(
        test_ids, tile_species, all_cat_ids
    )

    # L1 distances from global distribution
    d_train = np.sum(np.abs(p_train - global_proportions))
    d_val = np.sum(np.abs(p_val - global_proportions))
    d_test = np.sum(np.abs(p_test - global_proportions))

    distribution_score = d_train + d_val + d_test

    # Missing species penalty for val/test
    missing_penalty = 0.0
    missing_val = []
    missing_test = []

    for cat_id in all_cat_ids:
        global_count = global_counts.get(cat_id, 0)
        if global_count == 0:
            continue

        # Weight by global frequency (rare species matter more per-instance)
        # But also scale by sqrt to not over-penalize common species
        weight = np.sqrt(global_count / total_annotations)

        if counts_val.get(cat_id, 0) == 0:
            missing_penalty += weight
            missing_val.append(cat_id)
        if counts_test.get(cat_id, 0) == 0:
            missing_penalty += weight
            missing_test.append(cat_id)

    missing_penalty *= missing_penalty_weight

    # Annotation imbalance penalty
    # Expected: 80/10/10
    expected_train = 0.8 * total_annotations
    expected_val = 0.1 * total_annotations
    expected_test = 0.1 * total_annotations

    imbalance = (
        abs(total_train - expected_train) / total_annotations +
        abs(total_val - expected_val) / total_annotations +
        abs(total_test - expected_test) / total_annotations
    )
    imbalance_penalty = imbalance * imbalance_penalty_weight

    total_score = distribution_score + missing_penalty + imbalance_penalty

    details = {
        "distribution_score": distribution_score,
        "d_train": d_train,
        "d_val": d_val,
        "d_test": d_test,
        "missing_penalty": missing_penalty,
        "missing_val": missing_val,
        "missing_test": missing_test,
        "imbalance_penalty": imbalance_penalty,
        "total_train": total_train,
        "total_val": total_val,
        "total_test": total_test,
        "counts_train": counts_train,
        "counts_val": counts_val,
        "counts_test": counts_test,
    }

    return total_score, details


def generate_random_split(
    all_image_ids: List[int],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> Tuple[List[int], List[int], List[int]]:
    """Generate a random train/val/test split."""
    shuffled = all_image_ids.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_ids = shuffled[:train_end]
    val_ids = shuffled[train_end:val_end]
    test_ids = shuffled[val_end:]

    return train_ids, val_ids, test_ids


def find_best_split(
    all_image_ids: List[int],
    tile_species: Dict[int, Dict[int, int]],
    all_cat_ids: List[int],
    global_proportions: np.ndarray,
    global_counts: Dict[int, int],
    total_annotations: int,
    iterations: int = 3000,
    seed: int = 42,
    verbose: bool = True
) -> Tuple[List[int], List[int], List[int], float, Dict[str, Any]]:
    """
    Find the best split through random search.

    Returns:
        best_train, best_val, best_test, best_score, best_details
    """
    random.seed(seed)
    np.random.seed(seed)

    best_score = float("inf")
    best_split = None
    best_details = None

    scores_history = []

    for i in range(iterations):
        train_ids, val_ids, test_ids = generate_random_split(all_image_ids)

        score, details = score_split(
            train_ids, val_ids, test_ids,
            tile_species, all_cat_ids,
            global_proportions, global_counts, total_annotations
        )

        scores_history.append(score)

        if score < best_score:
            best_score = score
            best_split = (train_ids, val_ids, test_ids)
            best_details = details

            if verbose and (i < 10 or i % 500 == 0):
                print(f"  Iteration {i}: New best score = {score:.4f}")

    if verbose:
        print(f"\n  Search complete. Best score: {best_score:.4f}")
        print(f"  Mean score: {np.mean(scores_history):.4f}, Std: {np.std(scores_history):.4f}")

    return best_split[0], best_split[1], best_split[2], best_score, best_details


def create_split_annotations(
    image_ids: set,
    images: List[Dict],
    annotations: List[Dict],
    categories: List[Dict]
) -> Dict[str, Any]:
    """Create a COCO annotation dict for a subset of images."""
    # Filter images
    split_images = [img for img in images if img["id"] in image_ids]

    # Renumber images starting from 1
    old_to_new_id = {img["id"]: i + 1 for i, img in enumerate(split_images)}

    # Update image IDs
    new_images = []
    for img in split_images:
        new_img = img.copy()
        new_img["id"] = old_to_new_id[img["id"]]
        new_images.append(new_img)

    # Filter and update annotations
    new_annotations = []
    ann_id = 1
    for ann in annotations:
        if ann["image_id"] in image_ids:
            new_ann = ann.copy()
            new_ann["id"] = ann_id
            new_ann["image_id"] = old_to_new_id[ann["image_id"]]
            new_annotations.append(new_ann)
            ann_id += 1

    return {
        "images": new_images,
        "annotations": new_annotations,
        "categories": categories  # Keep full category list
    }


def generate_report(
    global_counts: Dict[int, int],
    best_details: Dict[str, Any],
    cat_id_to_name: Dict[int, str],
    all_cat_ids: List[int],
    train_ids: List[int],
    val_ids: List[int],
    test_ids: List[int],
    best_score: float,
    iterations: int,
    seed: int
) -> str:
    """Generate a diagnostic report."""
    lines = []
    lines.append("=" * 70)
    lines.append("DATASET SPLIT REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Configuration
    lines.append("CONFIGURATION")
    lines.append("-" * 40)
    lines.append(f"Random seed: {seed}")
    lines.append(f"Iterations: {iterations}")
    lines.append(f"Split ratios: 80% / 10% / 10%")
    lines.append(f"Final score: {best_score:.4f}")
    lines.append("")

    # Score breakdown
    lines.append("SCORE BREAKDOWN")
    lines.append("-" * 40)
    lines.append(f"Distribution score: {best_details['distribution_score']:.4f}")
    lines.append(f"  - Train L1 distance: {best_details['d_train']:.4f}")
    lines.append(f"  - Val L1 distance: {best_details['d_val']:.4f}")
    lines.append(f"  - Test L1 distance: {best_details['d_test']:.4f}")
    lines.append(f"Missing species penalty: {best_details['missing_penalty']:.4f}")
    lines.append(f"Imbalance penalty: {best_details['imbalance_penalty']:.4f}")
    lines.append("")

    # Split sizes
    lines.append("SPLIT SIZES")
    lines.append("-" * 40)
    total_tiles = len(train_ids) + len(val_ids) + len(test_ids)
    total_ann = best_details['total_train'] + best_details['total_val'] + best_details['total_test']

    lines.append(f"{'Split':<10} {'Tiles':>10} {'%':>8} {'Annotations':>15} {'%':>8}")
    lines.append("-" * 55)
    lines.append(f"{'Train':<10} {len(train_ids):>10} {100*len(train_ids)/total_tiles:>7.1f}% {best_details['total_train']:>15} {100*best_details['total_train']/total_ann:>7.1f}%")
    lines.append(f"{'Val':<10} {len(val_ids):>10} {100*len(val_ids)/total_tiles:>7.1f}% {best_details['total_val']:>15} {100*best_details['total_val']/total_ann:>7.1f}%")
    lines.append(f"{'Test':<10} {len(test_ids):>10} {100*len(test_ids)/total_tiles:>7.1f}% {best_details['total_test']:>15} {100*best_details['total_test']/total_ann:>7.1f}%")
    lines.append("-" * 55)
    lines.append(f"{'Total':<10} {total_tiles:>10} {'100.0%':>8} {total_ann:>15} {'100.0%':>8}")
    lines.append("")

    # Per-species distribution
    lines.append("PER-SPECIES DISTRIBUTION")
    lines.append("-" * 90)
    lines.append(f"{'Species':<12} {'Global':>8} {'Train':>8} {'Val':>8} {'Test':>8} {'Train%':>8} {'Val%':>8} {'Test%':>8}")
    lines.append("-" * 90)

    counts_train = best_details['counts_train']
    counts_val = best_details['counts_val']
    counts_test = best_details['counts_test']

    # Sort by global count descending
    sorted_cats = sorted(all_cat_ids, key=lambda x: -global_counts.get(x, 0))

    imbalance_scores = []

    for cat_id in sorted_cats:
        name = cat_id_to_name[cat_id][:12]
        g = global_counts.get(cat_id, 0)
        tr = counts_train.get(cat_id, 0)
        va = counts_val.get(cat_id, 0)
        te = counts_test.get(cat_id, 0)

        if g > 0:
            tr_pct = 100 * tr / g
            va_pct = 100 * va / g
            te_pct = 100 * te / g
            # Compute imbalance as deviation from 80/10/10
            imbalance = abs(tr_pct - 80) + abs(va_pct - 10) + abs(te_pct - 10)
            imbalance_scores.append((cat_id, imbalance, g))
        else:
            tr_pct = va_pct = te_pct = 0

        lines.append(f"{name:<12} {g:>8} {tr:>8} {va:>8} {te:>8} {tr_pct:>7.1f}% {va_pct:>7.1f}% {te_pct:>7.1f}%")

    lines.append("")

    # Missing species
    lines.append("MISSING SPECIES")
    lines.append("-" * 40)
    if best_details['missing_val']:
        lines.append(f"Missing in val ({len(best_details['missing_val'])}):")
        for cat_id in best_details['missing_val']:
            lines.append(f"  - {cat_id_to_name[cat_id]} (global count: {global_counts.get(cat_id, 0)})")
    else:
        lines.append("Missing in val: None")

    if best_details['missing_test']:
        lines.append(f"Missing in test ({len(best_details['missing_test'])}):")
        for cat_id in best_details['missing_test']:
            lines.append(f"  - {cat_id_to_name[cat_id]} (global count: {global_counts.get(cat_id, 0)})")
    else:
        lines.append("Missing in test: None")
    lines.append("")

    # Top 10 most imbalanced species (excluding those with very low counts)
    lines.append("TOP 10 MOST IMBALANCED SPECIES")
    lines.append("-" * 40)
    # Filter to species with at least 10 samples for meaningful comparison
    meaningful_imbalances = [(c, i, g) for c, i, g in imbalance_scores if g >= 10]
    meaningful_imbalances.sort(key=lambda x: -x[1])

    for cat_id, imbalance, g in meaningful_imbalances[:10]:
        name = cat_id_to_name[cat_id]
        tr = counts_train.get(cat_id, 0)
        va = counts_val.get(cat_id, 0)
        te = counts_test.get(cat_id, 0)
        lines.append(f"  {name}: {tr}/{va}/{te} (global: {g}, deviation: {imbalance:.1f})")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Find optimal train/val/test splits for bird detection dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python find_best_splits.py \\
        --tiles_dir data/BirdDataset_2025_nonoverlapping_tiles_500 \\
        --annotations data/BirdDataset_2025_nonoverlapping_tiles_500/all_annotations.json \\
        --iterations 3000
        """
    )
    parser.add_argument(
        "-t", "--tiles_dir",
        required=True,
        help="Directory containing the tiled images"
    )
    parser.add_argument(
        "-a", "--annotations",
        required=True,
        help="Path to COCO-style annotations file (all_annotations.json)"
    )
    parser.add_argument(
        "-n", "--iterations",
        type=int,
        default=3000,
        help="Number of random splits to try (default: 3000)"
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory for splits (default: <tiles_dir>/splits)"
    )

    args = parser.parse_args()

    # Set output directory
    output_dir = Path(args.output) if args.output else Path(args.tiles_dir) / "splits"

    print("=" * 60)
    print("FIND BEST DATASET SPLITS")
    print("=" * 60)
    print(f"Tiles directory: {args.tiles_dir}")
    print(f"Annotations: {args.annotations}")
    print(f"Iterations: {args.iterations}")
    print(f"Seed: {args.seed}")
    print(f"Output: {output_dir}")

    # Load data
    print("\nLoading annotations...")
    data = load_annotations(args.annotations)
    images = data["images"]
    annotations = data["annotations"]
    categories = data["categories"]

    # Print dataset summary
    print_dataset_summary(images, annotations, categories)

    # Compute tile-level species counts
    print("Computing per-tile species counts...")
    tile_species, cat_id_to_name, all_cat_ids = compute_tile_species_counts(
        annotations, categories
    )

    # Compute global distribution
    global_proportions, global_counts = compute_global_distribution(
        annotations, all_cat_ids
    )
    total_annotations = len(annotations)

    # Get all image IDs
    all_image_ids = [img["id"] for img in images]

    # Find best split
    print(f"\nSearching for optimal split ({args.iterations} iterations)...")
    print("-" * 40)

    train_ids, val_ids, test_ids, best_score, best_details = find_best_split(
        all_image_ids,
        tile_species,
        all_cat_ids,
        global_proportions,
        global_counts,
        total_annotations,
        iterations=args.iterations,
        seed=args.seed
    )

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate report
    print("\n" + "=" * 60)
    print("GENERATING OUTPUTS")
    print("=" * 60)

    report = generate_report(
        global_counts, best_details, cat_id_to_name, all_cat_ids,
        train_ids, val_ids, test_ids, best_score,
        args.iterations, args.seed
    )

    # Print report
    print("\n" + report)

    # Save report
    report_path = output_dir / "split_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to: {report_path}")

    # Create image ID to image mapping
    id_to_image = {img["id"]: img for img in images}

    # Save tile lists
    print("\nSaving tile lists...")
    for split_name, split_ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        tiles_path = output_dir / f"{split_name}_tiles.txt"
        with open(tiles_path, "w") as f:
            for img_id in sorted(split_ids):
                f.write(id_to_image[img_id]["file_name"] + "\n")
        print(f"  {split_name}_tiles.txt: {len(split_ids)} tiles")

    # Save COCO annotation files
    print("\nSaving COCO annotation files...")
    for split_name, split_ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        split_annotations = create_split_annotations(
            set(split_ids), images, annotations, categories
        )
        ann_path = output_dir / f"{split_name}_annotations.json"
        with open(ann_path, "w") as f:
            json.dump(split_annotations, f, indent=2)
        print(f"  {split_name}_annotations.json: {len(split_annotations['images'])} images, {len(split_annotations['annotations'])} annotations")

    # Save split metadata
    print("\nSaving split metadata...")
    metadata = {
        "seed": args.seed,
        "iterations": args.iterations,
        "best_score": best_score,
        "score_details": {
            "distribution_score": best_details["distribution_score"],
            "missing_penalty": best_details["missing_penalty"],
            "imbalance_penalty": best_details["imbalance_penalty"],
        },
        "split_sizes": {
            "train_tiles": len(train_ids),
            "val_tiles": len(val_ids),
            "test_tiles": len(test_ids),
            "train_annotations": best_details["total_train"],
            "val_annotations": best_details["total_val"],
            "test_annotations": best_details["total_test"],
        },
        "missing_species": {
            "val": [cat_id_to_name[c] for c in best_details["missing_val"]],
            "test": [cat_id_to_name[c] for c in best_details["missing_test"]],
        },
        "train_image_ids": sorted(train_ids),
        "val_image_ids": sorted(val_ids),
        "test_image_ids": sorted(test_ids),
    }

    metadata_path = output_dir / "split_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  split_metadata.json")

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print()


if __name__ == "__main__":
    main()
