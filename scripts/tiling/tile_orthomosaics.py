"""
Tile orthomosaic images into smaller patches for CO-DETR training.

Converts orthomosaic-level annotations to tile-local COCO format.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def compute_bbox_intersection(bbox_xyxy: list, tile_x0: int, tile_y0: int, tile_size: int) -> Optional[list]:
    """
    Compute intersection of bbox with tile window.

    Args:
        bbox_xyxy: [x_min, y_min, x_max, y_max] in orthomosaic coords
        tile_x0, tile_y0: tile upper-left corner
        tile_size: tile dimension

    Returns:
        Clipped bbox in tile-local coords as [x, y, w, h] or None if no intersection
    """
    x_min, y_min, x_max, y_max = bbox_xyxy

    tile_x1 = tile_x0 + tile_size
    tile_y1 = tile_y0 + tile_size

    # Compute intersection
    inter_x_min = max(x_min, tile_x0)
    inter_y_min = max(y_min, tile_y0)
    inter_x_max = min(x_max, tile_x1)
    inter_y_max = min(y_max, tile_y1)

    # Check if valid intersection
    if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
        return None

    # Convert to tile-local coordinates
    local_x_min = inter_x_min - tile_x0
    local_y_min = inter_y_min - tile_y0
    local_x_max = inter_x_max - tile_x0
    local_y_max = inter_y_max - tile_y0

    # Convert to xywh format for COCO
    w = local_x_max - local_x_min
    h = local_y_max - local_y_min

    return [local_x_min, local_y_min, w, h]


def compute_overlap_ratio(bbox_xyxy: list, tile_x0: int, tile_y0: int, tile_size: int) -> float:
    """
    Compute what fraction of the original bbox is inside the tile.
    """
    x_min, y_min, x_max, y_max = bbox_xyxy
    original_area = (x_max - x_min) * (y_max - y_min)

    if original_area <= 0:
        return 0.0

    tile_x1 = tile_x0 + tile_size
    tile_y1 = tile_y0 + tile_size

    inter_x_min = max(x_min, tile_x0)
    inter_y_min = max(y_min, tile_y0)
    inter_x_max = min(x_max, tile_x1)
    inter_y_max = min(y_max, tile_y1)

    if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    return inter_area / original_area


def clip_segmentation_to_tile(
    segmentation: list,
    tile_x0: int,
    tile_y0: int,
    tile_size: int
) -> list:
    """
    Convert segmentation polygon to tile-local coordinates.

    Simple approach: subtract tile origin, clip points to tile bounds.
    Note: This doesn't do proper polygon clipping, just point-wise clipping.
    For birds that are mostly inside the tile, this is usually acceptable.
    """
    if not segmentation or not segmentation[0]:
        return []

    # segmentation is [[x1, y1, x2, y2, ...]]
    poly = segmentation[0]
    local_poly = []

    for i in range(0, len(poly), 2):
        if i + 1 >= len(poly):
            break
        x = poly[i] - tile_x0
        y = poly[i + 1] - tile_y0

        # Clip to tile bounds
        x = max(0, min(x, tile_size))
        y = max(0, min(y, tile_size))

        local_poly.extend([x, y])

    return [local_poly] if local_poly else []


def tile_orthomosaic(
    image_path: Path,
    annotation_json_path: Path,
    output_root: Path,
    tile_size: int = 512,
    stride: int = 432,
    overlap_threshold: float = 0.8,
    jpeg_quality: int = 95
) -> dict:
    """
    Tile one orthomosaic and convert annotations to COCO format.

    Args:
        image_path: Path to orthomosaic image
        annotation_json_path: Path to annotation JSON for this orthomosaic
        output_root: Root output directory
        tile_size: Tile dimension (default 512)
        stride: Step between tiles (default 432 for ~15.6% overlap)
        overlap_threshold: Minimum bbox overlap ratio to include annotation (default 0.8)
        jpeg_quality: JPEG save quality (default 95)

    Returns:
        Summary dict with counts
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    img_height, img_width = img.shape[:2]

    # Load annotations
    with open(annotation_json_path, 'r') as f:
        ann_data = json.load(f)

    annotations = ann_data.get('annotations', [])

    # Collect all unique species codes to build category mapping
    species_codes = sorted(set(a['tcws_species'] for a in annotations if 'tcws_species' in a))
    species_to_cat_id = {sp: idx for idx, sp in enumerate(species_codes)}

    # Create output directory structure
    ortho_stem = image_path.stem
    output_dir = output_root / ortho_stem
    tiles_dir = output_dir / 'image_tiles'
    tiles_dir.mkdir(parents=True, exist_ok=True)

    # COCO output structure
    coco_images = []
    coco_annotations = []
    coco_categories = [{'id': idx, 'name': sp} for sp, idx in species_to_cat_id.items()]

    image_id = 0
    annotation_id = 0
    tiles_created = 0
    annotations_kept = 0

    # Generate tile positions
    tile_positions = []
    for y0 in range(0, img_height, stride):
        for x0 in range(0, img_width, stride):
            # Adjust if tile would extend beyond image
            if x0 + tile_size > img_width:
                x0 = max(0, img_width - tile_size)
            if y0 + tile_size > img_height:
                y0 = max(0, img_height - tile_size)

            tile_positions.append((x0, y0))

    # Remove duplicate positions from boundary adjustments
    tile_positions = list(set(tile_positions))
    tile_positions.sort(key=lambda p: (p[1], p[0]))  # Sort by y, then x

    for x0, y0 in tile_positions:
        image_id += 1

        # Extract tile
        tile = img[y0:y0 + tile_size, x0:x0 + tile_size]

        # Skip if tile is smaller than expected (edge case)
        if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
            continue

        # Find annotations that belong to this tile
        tile_anns = []
        for ann in annotations:
            bbox_xyxy = ann['bbox']

            # Check overlap ratio
            overlap = compute_overlap_ratio(bbox_xyxy, x0, y0, tile_size)
            if overlap < overlap_threshold:
                continue

            # Convert bbox to tile-local xywh
            local_bbox = compute_bbox_intersection(bbox_xyxy, x0, y0, tile_size)
            if local_bbox is None:
                continue

            # Convert segmentation to tile-local
            local_seg = clip_segmentation_to_tile(
                ann.get('segmentation', []), x0, y0, tile_size
            )

            # Get category ID
            species = ann.get('tcws_species', 'UNKNOWN')
            cat_id = species_to_cat_id.get(species, 0)

            annotation_id += 1
            tile_ann = {
                'id': annotation_id,
                'image_id': image_id,
                'category_id': cat_id,
                'bbox': [round(v, 2) for v in local_bbox],
                'area': round(local_bbox[2] * local_bbox[3], 2),
                'iscrowd': 0,
                'ignore': 0
            }

            if local_seg:
                tile_ann['segmentation'] = [[round(v, 2) for v in local_seg[0]]]

            tile_anns.append(tile_ann)

        # Save tile image
        tile_filename = f"{ortho_stem}_{image_id:05d}_x{x0}_y{y0}.jpg"
        tile_path = tiles_dir / tile_filename
        cv2.imwrite(str(tile_path), tile, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

        # Add image entry
        coco_images.append({
            'id': image_id,
            'file_name': f"image_tiles/{tile_filename}",
            'width': tile_size,
            'height': tile_size
        })

        coco_annotations.extend(tile_anns)
        tiles_created += 1
        annotations_kept += len(tile_anns)

    # Write COCO JSON
    coco_output = {
        'images': coco_images,
        'annotations': coco_annotations,
        'categories': coco_categories
    }

    output_json_path = output_dir / 'annotations.json'
    with open(output_json_path, 'w') as f:
        json.dump(coco_output, f, indent=2)

    return {
        'orthomosaic': ortho_stem,
        'tiles_created': tiles_created,
        'annotations_total': len(annotations),
        'annotations_kept': annotations_kept,
        'categories': len(coco_categories),
        'output_dir': str(output_dir)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Tile orthomosaic images and convert annotations to COCO format'
    )
    parser.add_argument(
        '--image', '-i',
        type=Path,
        required=True,
        help='Path to orthomosaic image'
    )
    parser.add_argument(
        '--annotations', '-a',
        type=Path,
        required=True,
        help='Path to annotation JSON for the orthomosaic'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        required=True,
        help='Output root directory'
    )
    parser.add_argument(
        '--tile-size',
        type=int,
        default=512,
        help='Tile size in pixels (default: 512)'
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=432,
        help='Stride between tiles (default: 432 for ~15.6%% overlap)'
    )
    parser.add_argument(
        '--overlap-threshold',
        type=float,
        default=0.8,
        help='Minimum bbox overlap ratio to include annotation (default: 0.8)'
    )
    parser.add_argument(
        '--jpeg-quality',
        type=int,
        default=95,
        help='JPEG save quality (default: 95)'
    )

    args = parser.parse_args()

    result = tile_orthomosaic(
        image_path=args.image,
        annotation_json_path=args.annotations,
        output_root=args.output,
        tile_size=args.tile_size,
        stride=args.stride,
        overlap_threshold=args.overlap_threshold,
        jpeg_quality=args.jpeg_quality
    )

    print(f"Processed: {result['orthomosaic']}")
    print(f"  Tiles created: {result['tiles_created']}")
    print(f"  Annotations: {result['annotations_kept']}/{result['annotations_total']} kept")
    print(f"  Categories: {result['categories']}")
    print(f"  Output: {result['output_dir']}")


if __name__ == '__main__':
    main()
