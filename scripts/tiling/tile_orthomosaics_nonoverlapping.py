"""
Tile a single orthomosaic image into non-overlapping patches.

Converts orthomosaic-level annotations to COCO format with tile-local coordinates.

Key features:
- Configurable tile size (default 512, use 500 for zero-waste on 10k images)
- Non-overlapping tiles, drops incomplete edge tiles
- Keeps partial/clipped bounding boxes
- Tile naming: OM_{id}_{x:05d}_{y:05d}.png

Usage:
    # Tile a single orthomosaic with 512x512 tiles
    cd scripts/tiling/
    python tile_orthomosaics_nonoverlapping.py \\
        -i /path/to/OM_001_20250510_10k.jpg \\
        -a /path/to/OM_001_annotations.json \\
        -o /path/to/output/

    # Use 500x500 tiles (no waste on 10k images)
    cd scripts/tiling/
    python tile_orthomosaics_nonoverlapping.py \\
        -i /path/to/OM_001.jpg \\
        -a /path/to/OM_001_annotations.json \\
        -o /path/to/output/ \\
        --tile-size 500
"""

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MIN_BBOX_AREA = 4.0  # Minimum area in pixels to keep a clipped bbox


@dataclass
class TilingStats:
    """Statistics for tiling process."""
    orthomosaic_id: str
    image_width: int
    image_height: int
    tile_size: int
    total_tiles: int = 0
    dropped_edge_tiles: int = 0
    total_annotations: int = 0
    clipped_boxes: int = 0
    empty_tiles: int = 0
    tiles_with_partial_boxes: int = 0

    def to_dict(self) -> dict:
        return {
            'orthomosaic_id': self.orthomosaic_id,
            'image_size': f"{self.image_width}x{self.image_height}",
            'tile_size': self.tile_size,
            'total_tiles': self.total_tiles,
            'dropped_edge_tiles': self.dropped_edge_tiles,
            'total_annotations': self.total_annotations,
            'clipped_boxes': self.clipped_boxes,
            'empty_tiles': self.empty_tiles,
            'tiles_with_partial_boxes': self.tiles_with_partial_boxes,
        }


def extract_om_id(filename: str) -> str:
    """
    Extract orthomosaic ID from filename.

    Assumes filenames follow the pattern: OM_XXX_...
    Example: 'OM_001_20250510_10k-03-11.jpg' -> 'OM_001'
    """
    match = re.match(r'(OM_\d{3})', filename)
    if match:
        return match.group(1)

    # Fallback: use stem without extension
    stem = Path(filename).stem
    logger.warning(f"Could not extract OM_XXX pattern from {filename}, using stem: {stem}")
    return stem


def compute_bbox_intersection(
    bbox_xyxy: list,
    tile_x0: int,
    tile_y0: int,
    tile_size: int
) -> Optional[tuple]:
    """
    Compute intersection of bbox with tile window.

    Args:
        bbox_xyxy: [x_min, y_min, x_max, y_max] in orthomosaic coords
        tile_x0, tile_y0: tile upper-left corner
        tile_size: tile dimension

    Returns:
        Tuple of (local_bbox_xywh, is_clipped) or (None, False) if no intersection
        local_bbox_xywh: [x, y, w, h] in tile-local coordinates
    """
    x_min, y_min, x_max, y_max = bbox_xyxy

    tile_x1 = tile_x0 + tile_size
    tile_y1 = tile_y0 + tile_size

    # Check if bbox intersects tile at all
    if x_max <= tile_x0 or x_min >= tile_x1:
        return None, False
    if y_max <= tile_y0 or y_min >= tile_y1:
        return None, False

    # Compute intersection (clip to tile bounds)
    inter_x_min = max(x_min, tile_x0)
    inter_y_min = max(y_min, tile_y0)
    inter_x_max = min(x_max, tile_x1)
    inter_y_max = min(y_max, tile_y1)

    # Check if clipped
    is_clipped = (
        inter_x_min > x_min or
        inter_y_min > y_min or
        inter_x_max < x_max or
        inter_y_max < y_max
    )

    # Convert to tile-local coordinates
    local_x = inter_x_min - tile_x0
    local_y = inter_y_min - tile_y0
    local_w = inter_x_max - inter_x_min
    local_h = inter_y_max - inter_y_min

    # Check minimum area
    if local_w * local_h < MIN_BBOX_AREA:
        return None, False

    return [local_x, local_y, local_w, local_h], is_clipped


def build_category_mapping(detections: list) -> tuple:
    """
    Build category ID mapping from detections.

    Returns:
        (species_to_id dict, coco_categories list)
    """
    species_set = set()
    for det in detections:
        species = det.get('tcws_species', 'UNKNOWN')
        species_set.add(species)

    # Sort for reproducibility
    species_list = sorted(species_set)
    species_to_id = {sp: idx + 1 for idx, sp in enumerate(species_list)}

    coco_categories = [
        {'id': idx + 1, 'name': sp}
        for idx, sp in enumerate(species_list)
    ]

    return species_to_id, coco_categories


def tile_orthomosaic(
    image_path: Path,
    annotation_path: Path,
    output_dir: Path,
    tile_size: int = 512,
) -> dict:
    """
    Tile a single orthomosaic image and create COCO annotations.

    Args:
        image_path: Path to orthomosaic image file
        annotation_path: Path to annotation JSON file for this orthomosaic
        output_dir: Output directory for tiles and COCO JSON
        tile_size: Size of tiles in pixels (default 512)

    Returns:
        Summary statistics dict
    """
    # Extract OM ID from filename
    om_id = extract_om_id(image_path.name)
    logger.info(f"Processing {om_id} from {image_path.name}")

    # Load annotations
    logger.info(f"Loading annotations from {annotation_path}")
    with open(annotation_path, 'r') as f:
        ann_data = json.load(f)

    # Handle different annotation formats
    # Format 1: {"annotations": [...]} (per-orthomosaic file)
    # Format 2: {"images": [{"file_name": ..., "detections": [...]}]} (all_annotations.json)
    if 'annotations' in ann_data:
        detections = ann_data['annotations']
    elif 'images' in ann_data:
        # Find the matching image entry
        target_filename = image_path.name
        detections = []
        for img_entry in ann_data['images']:
            if img_entry['file_name'] == target_filename:
                detections = img_entry.get('detections', [])
                break
        if not detections:
            logger.warning(f"No detections found for {target_filename} in annotation file")
    else:
        raise ValueError(f"Unknown annotation format in {annotation_path}")

    logger.info(f"Found {len(detections)} detections")

    # Build category mapping
    species_to_id, coco_categories = build_category_mapping(detections)
    logger.info(f"Found {len(coco_categories)} species categories")
    logger.info(f"Using tile size: {tile_size}x{tile_size}")

    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    img_height, img_width = img.shape[:2]
    logger.info(f"Image size: {img_width}x{img_height}")

    # Initialize stats
    stats = TilingStats(
        orthomosaic_id=om_id,
        image_width=img_width,
        image_height=img_height,
        tile_size=tile_size
    )

    # Calculate grid dimensions (only complete tiles)
    n_tiles_x = img_width // tile_size
    n_tiles_y = img_height // tile_size

    # Calculate dropped edge pixels
    remainder_x = img_width % tile_size
    remainder_y = img_height % tile_size

    if remainder_x > 0:
        stats.dropped_edge_tiles += n_tiles_y
        logger.info(f"Dropping {remainder_x}px on right edge ({n_tiles_y} partial tiles)")
    if remainder_y > 0:
        stats.dropped_edge_tiles += n_tiles_x
        logger.info(f"Dropping {remainder_y}px on bottom edge ({n_tiles_x} partial tiles)")
    if remainder_x > 0 and remainder_y > 0:
        stats.dropped_edge_tiles += 1  # Corner tile

    # Create output directories
    tiles_dir = output_dir / 'tiles'
    tiles_dir.mkdir(parents=True, exist_ok=True)

    coco_images = []
    coco_annotations = []
    image_id = 0
    annotation_id = 0

    # Process each complete tile
    total_tiles = n_tiles_x * n_tiles_y
    logger.info(f"Generating {n_tiles_x}x{n_tiles_y} = {total_tiles} tiles")

    for tile_y_idx in tqdm(range(n_tiles_y), desc="Processing rows"):
        for tile_x_idx in range(n_tiles_x):
            tile_x0 = tile_x_idx * tile_size
            tile_y0 = tile_y_idx * tile_size

            image_id += 1
            stats.total_tiles += 1

            # Extract tile
            tile = img[tile_y0:tile_y0 + tile_size, tile_x0:tile_x0 + tile_size]

            # Generate tile filename with zero-padded coordinates
            tile_filename = f"{om_id}_{tile_x0:05d}_{tile_y0:05d}.png"
            tile_path = tiles_dir / tile_filename

            # Find annotations for this tile
            tile_annotations = []
            has_partial_box = False

            for det in detections:
                bbox_xyxy = det['bbox']
                local_bbox, is_clipped = compute_bbox_intersection(
                    bbox_xyxy, tile_x0, tile_y0, tile_size
                )

                if local_bbox is None:
                    continue

                if is_clipped:
                    stats.clipped_boxes += 1
                    has_partial_box = True

                species = det.get('tcws_species', 'UNKNOWN')
                category_id = species_to_id.get(species, 0)

                annotation_id += 1
                ann = {
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': category_id,
                    'bbox': [round(v, 2) for v in local_bbox],
                    'area': round(local_bbox[2] * local_bbox[3], 2),
                    'iscrowd': 0,
                }
                tile_annotations.append(ann)

            if has_partial_box:
                stats.tiles_with_partial_boxes += 1

            if len(tile_annotations) == 0:
                stats.empty_tiles += 1

            stats.total_annotations += len(tile_annotations)

            # Save tile image
            cv2.imwrite(str(tile_path), tile)

            # Add COCO image entry
            coco_images.append({
                'id': image_id,
                'file_name': tile_filename,
                'width': tile_size,
                'height': tile_size,
                'orthomosaic_id': om_id,
                'tile_x': tile_x0,
                'tile_y': tile_y0,
            })

            coco_annotations.extend(tile_annotations)

    # Build COCO dataset (categories before annotations)
    coco_dataset = {
        'info': {
            'description': f'Seabird Detection Dataset - {om_id} Tiled',
            'version': '1.0',
            'tile_size': tile_size,
            'source_image': image_path.name,
            'source_dimensions': f"{img_width}x{img_height}",
        },
        'categories': coco_categories,
        'images': coco_images,
        'annotations': coco_annotations,
    }

    # Save COCO JSON
    coco_json_path = output_dir / 'annotations.json'
    logger.info(f"Saving COCO annotations to {coco_json_path}")
    with open(coco_json_path, 'w') as f:
        json.dump(coco_dataset, f, indent=2)

    # Save diagnostics
    diagnostics = stats.to_dict()
    diagnostics_path = output_dir / 'tiling_diagnostics.json'
    with open(diagnostics_path, 'w') as f:
        json.dump(diagnostics, f, indent=2)

    return diagnostics


def main():
    parser = argparse.ArgumentParser(
        description='Tile a single orthomosaic image into non-overlapping patches'
    )
    parser.add_argument(
        '--image', '-i',
        type=Path,
        required=True,
        help='Path to orthomosaic image file (e.g., OM_001_20250510_10k.jpg)'
    )
    parser.add_argument(
        '--annotations', '-a',
        type=Path,
        required=True,
        help='Path to annotation JSON file for this orthomosaic'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        required=True,
        help='Output directory for tiles and annotations'
    )
    parser.add_argument(
        '--tile-size', '-t',
        type=int,
        default=512,
        help='Tile size in pixels (default: 512). Use 500 for zero-waste on 10k images.'
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.image.exists():
        raise FileNotFoundError(f"Image file not found: {args.image}")
    if not args.annotations.exists():
        raise FileNotFoundError(f"Annotations file not found: {args.annotations}")
    if args.tile_size <= 0:
        raise ValueError(f"Tile size must be positive, got: {args.tile_size}")

    summary = tile_orthomosaic(
        image_path=args.image,
        annotation_path=args.annotations,
        output_dir=args.output,
        tile_size=args.tile_size,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("TILING COMPLETE")
    print("=" * 60)
    print(f"Orthomosaic: {summary['orthomosaic_id']}")
    print(f"Image size: {summary['image_size']}")
    print(f"Tile size: {summary['tile_size']}x{summary['tile_size']}")
    print(f"Total tiles created: {summary['total_tiles']}")
    print(f"Edge tiles dropped: {summary['dropped_edge_tiles']}")
    print(f"Total annotations: {summary['total_annotations']}")
    print(f"Clipped boxes: {summary['clipped_boxes']}")
    print(f"Empty tiles: {summary['empty_tiles']}")
    print(f"Tiles with partial boxes: {summary['tiles_with_partial_boxes']}")
    print("=" * 60)


if __name__ == '__main__':
    main()
