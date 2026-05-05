#!/usr/bin/env python3
"""
Prepare a BirdDataset_2025_10k_MMDDYYYY folder from a source images directory
and an all-annotations JSON file.

Usage:
    python prepare_bird_dataset.py <images_dir> <annotations_json> [--output-dir DIR] [--samples-dir DIR]

Arguments:
    images_dir        Directory containing source orthophoto images
                      (files matching "20250510 10k-*.jpg" are treated as orthophotos;
                       DJI_* files are ignored)
    annotations_json  Path to the all-annotations JSON file (hank-updated-json.json format)

Options:
    --output-dir DIR  Where to create the dataset folder (default: same parent as images_dir)
    --samples-dir DIR Directory of sample/reference files to copy into samples/ (optional)
"""

import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from datetime import date


# ── Helpers ──────────────────────────────────────────────────────────────────

def is_orthophoto(filename: str) -> bool:
    """Return True for orthophoto tiles, False for DJI/other reference images."""
    return bool(re.match(r"^\d{8} 10k-\d+-\d+\.jpg$", filename))


def tile_sort_key(filename: str) -> tuple:
    """Sort key: extract (row, col) integers from "20250510 10k-RR-CC.jpg"."""
    m = re.search(r"10k-(\d+)-(\d+)", filename)
    return (int(m.group(1)), int(m.group(2))) if m else (999, 999)


def new_image_name(index: int, original_filename: str) -> str:
    """OM_NNN_20250510_10k-RR-CC.jpg"""
    tile_part = original_filename.split(" ", 1)[1]          # "10k-RR-CC.jpg"
    date_part = original_filename.split(" ", 1)[0]          # "20250510"
    return f"OM_{index:03d}_{date_part}_{tile_part}"


# ── Main ─────────────────────────────────────────────────────────────────────

def build_dataset(images_dir: str, annotations_json: str,
                  output_dir: str | None = None, samples_dir: str | None = None):

    # ── Resolve output path ───────────────────────────────────────────────
    today = date.today().strftime("%m%d%Y")
    dataset_name = f"BirdDataset_2025_10k_{today}"
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(images_dir))
    out = os.path.join(output_dir, dataset_name)

    if os.path.exists(out):
        raise FileExistsError(
            f"Output directory already exists: {out}\n"
            "Delete it or choose a different --output-dir."
        )

    # ── Load annotations ──────────────────────────────────────────────────
    with open(annotations_json) as f:
        data = json.load(f)

    anno_by_filename = {img["file_name"]: img for img in data["images"]}

    # ── Collect and sort orthophotos ──────────────────────────────────────
    source_files = sorted(
        [f for f in os.listdir(images_dir) if is_orthophoto(f)],
        key=tile_sort_key,
    )

    if not source_files:
        raise ValueError(f"No orthophoto files found in {images_dir}")

    missing = [f for f in source_files if f not in anno_by_filename]
    if missing:
        print(f"WARNING: {len(missing)} image(s) have no annotation entry and will have empty annotations:")
        for f in missing:
            print(f"  {f}")

    print(f"Found {len(source_files)} orthophotos → {dataset_name}")

    # ── Create directory structure ────────────────────────────────────────
    for subdir in ("images", "annotations", "samples", "metadata"):
        os.makedirs(os.path.join(out, subdir))

    # ── Copy images + build annotation files ─────────────────────────────
    all_anno_entries = []

    for i, src_name in enumerate(source_files, start=1):
        dst_name = new_image_name(i, src_name)

        # Copy image
        shutil.copy2(
            os.path.join(images_dir, src_name),
            os.path.join(out, "images", dst_name),
        )

        src_entry = anno_by_filename.get(src_name, {})
        detections = src_entry.get("detections", [])
        width  = src_entry.get("width",  10000)
        height = src_entry.get("height", 10000)

        # Per-image annotation JSON
        per_image = {
            "image": {"file_name": dst_name, "width": width, "height": height},
            "annotations": detections,
        }
        with open(os.path.join(out, "annotations", f"OM_{i:03d}_annotations.json"), "w") as f:
            json.dump(per_image, f, indent=2)

        all_anno_entries.append({
            "file_name": dst_name,
            "width": width,
            "height": height,
            "detections": detections,
        })

        print(f"  OM_{i:03d}  {src_name}  ({len(detections)} detections)")

    # ── all_annotations.json ──────────────────────────────────────────────
    with open(os.path.join(out, "annotations", "all_annotations.json"), "w") as f:
        json.dump({"images": all_anno_entries}, f, indent=2)

    # ── Copy samples ──────────────────────────────────────────────────────
    if samples_dir and os.path.isdir(samples_dir):
        for fname in os.listdir(samples_dir):
            shutil.copy2(
                os.path.join(samples_dir, fname),
                os.path.join(out, "samples", fname),
            )
        print(f"Copied samples from {samples_dir}")

    # ── Generate metadata summary ─────────────────────────────────────────
    all_species: set[str] = set()
    for entry in all_anno_entries:
        for det in entry["detections"]:
            all_species.add(det.get("tcws_species", "UNKN"))
    all_species_sorted = sorted(all_species)

    species_totals: dict[str, int] = defaultdict(int)
    rows = []
    for entry in all_anno_entries:
        sp_counts: dict[str, int] = defaultdict(int)
        for det in entry["detections"]:
            sp = det.get("tcws_species", "UNKN")
            sp_counts[sp] += 1
            species_totals[sp] += 1
        rows.append((entry["file_name"], len(entry["detections"]), sp_counts))

    total_birds = sum(species_totals.values())

    sp_header = "".join(f"  {sp:<6}" for sp in all_species_sorted)
    header = f"| {'Image Name':<28} | {'Masks':>7} |{sp_header} |"

    meta_lines = [
        f"Images loaded : {len(all_anno_entries)}",
        "JSonSys : CO-DETR - Master.json",
        "",
        "TCWS Species : ",
        header,
    ]
    for img_name, masks, sp_counts in rows:
        sp_vals = "".join(f"  {sp_counts.get(sp, 0):>6}" for sp in all_species_sorted)
        meta_lines.append(f"| {img_name:<28} | {masks:>7,} |{sp_vals} |")

    meta_lines += ["", "Species : "]
    for sp in sorted(species_totals, key=lambda s: -species_totals[s]):
        meta_lines.append(f"{sp:<8} : {species_totals[sp]:>10,} ")

    meta_lines += ["", f"Total birds : {total_birds:,}"]

    meta_filename = f"CO-DETR - Master Summary {date.today().strftime('%Y%m%d')}.txt"
    with open(os.path.join(out, "metadata", meta_filename), "w") as f:
        f.write("\n".join(meta_lines))

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\nDataset created at: {out}")
    print(f"  Images     : {len(all_anno_entries)}")
    print(f"  Species    : {len(species_totals)}")
    print(f"  Total birds: {total_birds:,}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare a BirdDataset_2025_10k_MMDDYYYY folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("images_dir",       help="Directory containing source orthophoto images")
    parser.add_argument("annotations_json", help="Path to all-annotations JSON file")
    parser.add_argument("--output-dir",     help="Where to create the dataset folder (default: parent of images_dir)")
    parser.add_argument("--samples-dir",    help="Directory of sample/reference files to copy into samples/")
    args = parser.parse_args()

    build_dataset(
        images_dir=args.images_dir,
        annotations_json=args.annotations_json,
        output_dir=args.output_dir,
        samples_dir=args.samples_dir,
    )


if __name__ == "__main__":
    main()
