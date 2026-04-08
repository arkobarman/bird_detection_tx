"""
Merge individual orthomosaic COCO annotations into one combined file.

This script reads all annotations.json files from orthomosaic subdirectories
and combines them into a single COCO-format JSON file with unique IDs.

Features:
- Collects annotations from all OM_*/annotations.json files
- Reassigns image and annotation IDs to be globally unique
- Remaps category IDs consistently across all orthomosaics
- Updates file paths to include subdirectory (e.g., OM_001/tiles/...)

Usage:
    cd scripts/tiling/
    python merge_annotations.py \\
        -i /path/to/tiled_output/ \\
        -o /path/to/tiled_output/all_annotations.json \\
        -t 500

    # Example with repository structure:
    python merge_annotations.py \\
        -i ../../data/tiled_500/ \\
        -o ../../data/tiled_500/all_annotations.json \\
        -t 500
"""

import argparse
import json
from pathlib import Path


def merge_annotations(input_dir: Path, output_path: Path, tile_size: int = 500):
    """
    Merge all annotations.json files from subdirectories into one combined file.

    Args:
        input_dir: Directory containing OM_XXX subdirectories with annotations.json
        output_path: Output path for combined annotations file
        tile_size: Tile size used (for info section)
    """
    all_images = []
    all_annotations = []
    all_categories = {}  # Use dict to deduplicate by name

    image_id_offset = 0
    annotation_id_offset = 0

    # Find all annotation files
    annotation_files = sorted(input_dir.glob('OM_*/annotations.json'))

    if not annotation_files:
        raise FileNotFoundError(f"No annotations.json files found in {input_dir}/OM_*/")

    print(f"Found {len(annotation_files)} annotation files to merge")

    for ann_file in annotation_files:
        om_id = ann_file.parent.name
        print(f"  Processing {om_id}...")

        with open(ann_file, 'r') as f:
            data = json.load(f)

        # Collect categories (deduplicate by name)
        for cat in data.get('categories', []):
            cat_name = cat['name']
            if cat_name not in all_categories:
                all_categories[cat_name] = cat

        # Build old_id -> new_id mapping for images
        image_id_map = {}
        for img in data.get('images', []):
            old_id = img['id']
            new_id = image_id_offset + old_id
            image_id_map[old_id] = new_id

            # Update image with new ID and prefixed filename
            new_img = img.copy()
            new_img['id'] = new_id
            # Update file_name to include OM subdirectory path
            new_img['file_name'] = f"{om_id}/tiles/{img['file_name']}"
            all_images.append(new_img)

        # Update annotations with new IDs
        for ann in data.get('annotations', []):
            annotation_id_offset += 1
            new_ann = ann.copy()
            new_ann['id'] = annotation_id_offset
            new_ann['image_id'] = image_id_map[ann['image_id']]
            all_annotations.append(new_ann)

        # Update offset for next file
        if data.get('images'):
            image_id_offset = max(img['id'] for img in all_images)

    # Reassign category IDs consistently
    sorted_categories = sorted(all_categories.keys())
    category_name_to_new_id = {name: idx + 1 for idx, name in enumerate(sorted_categories)}

    final_categories = [
        {'id': category_name_to_new_id[name], 'name': name}
        for name in sorted_categories
    ]

    # Update annotation category_ids
    # First build old category name -> old id mapping from each file
    # Actually we need to remap based on category names
    # For simplicity, let's rebuild category_id based on category name in annotations

    # We need the original category mappings - let's re-read to get name mappings
    # Actually, the current approach assumes category_id maps to same category across files
    # Let's fix this properly by tracking category name per annotation

    # Re-process to get category names
    print("  Remapping category IDs...")
    annotation_files = sorted(input_dir.glob('OM_*/annotations.json'))

    # Build per-file category_id -> name mapping
    file_cat_mappings = {}
    for ann_file in annotation_files:
        with open(ann_file, 'r') as f:
            data = json.load(f)
        cat_id_to_name = {cat['id']: cat['name'] for cat in data.get('categories', [])}
        file_cat_mappings[ann_file] = cat_id_to_name

    # Now update annotations with correct category IDs
    ann_idx = 0
    for ann_file in annotation_files:
        cat_id_to_name = file_cat_mappings[ann_file]
        with open(ann_file, 'r') as f:
            data = json.load(f)

        for ann in data.get('annotations', []):
            old_cat_id = ann['category_id']
            cat_name = cat_id_to_name.get(old_cat_id, 'UNKNOWN')
            new_cat_id = category_name_to_new_id.get(cat_name, 0)

            # Get the corresponding annotation from all_annotations
            all_annotations[ann_idx]['category_id'] = new_cat_id
            ann_idx += 1

    # Build combined COCO dataset
    combined = {
        'info': {
            'description': 'Seabird Detection Dataset - All Orthomosaics Tiled',
            'version': '1.0',
            'tile_size': tile_size,
            'num_orthomosaics': len(annotation_files),
        },
        'categories': final_categories,
        'images': all_images,
        'annotations': all_annotations,
    }

    # Save combined annotations
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(combined, f, indent=2)

    print(f"\nMerged annotations saved to: {output_path}")
    print(f"  Total images: {len(all_images)}")
    print(f"  Total annotations: {len(all_annotations)}")
    print(f"  Total categories: {len(final_categories)}")


def main():
    parser = argparse.ArgumentParser(
        description='Merge COCO annotations from multiple orthomosaics'
    )
    parser.add_argument(
        '--input', '-i',
        type=Path,
        required=True,
        help='Input directory containing OM_XXX subdirectories'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        required=True,
        help='Output path for combined annotations JSON'
    )
    parser.add_argument(
        '--tile-size', '-t',
        type=int,
        default=500,
        help='Tile size (for info section, default: 500)'
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input}")

    merge_annotations(
        input_dir=args.input,
        output_path=args.output,
        tile_size=args.tile_size,
    )


if __name__ == '__main__':
    main()
