"""
Deduplicate bird detections across overlapping tiles and generate crops.

This script:
1. Reads tiled COCO annotations and recovers global/orthomosaic coordinates
2. Detects duplicates using bbox overlap with union-find clustering
3. Merges duplicate clusters (same species) or flags conflicts (different species)
4. Crops deduplicated birds from the original orthomosaic (preferred) or tiles
"""

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class Annotation:
    """Single annotation with both tile-local and global coordinates."""
    ann_id: int
    image_id: int
    tile_file: str
    category_id: int
    category_name: str
    # Tile-local bbox (xywh)
    local_bbox_xywh: list
    # Global/orthomosaic bbox (xyxy)
    global_bbox_xyxy: list
    # Tile origin in orthomosaic coords
    tile_x0: int
    tile_y0: int


@dataclass
class DuplicateGroup:
    """A cluster of duplicate annotations."""
    group_id: int
    annotation_ids: list = field(default_factory=list)
    annotations: list = field(default_factory=list)
    category_id: int = -1
    category_name: str = ""
    # Merged bbox in global coords (xyxy)
    merged_bbox_xyxy: list = field(default_factory=list)


@dataclass
class Conflict:
    """Overlapping annotations with different species labels."""
    ann_id_1: int
    ann_id_2: int
    category_1: str
    category_2: str
    overlap_ratio_1: float
    overlap_ratio_2: float


# =============================================================================
# Union-Find for clustering
# =============================================================================

class UnionFind:
    """Union-Find data structure for clustering duplicates."""

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        # Union by rank
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

    def get_groups(self) -> dict:
        """Return dict mapping root -> list of members."""
        groups = defaultdict(list)
        for x in self.parent:
            groups[self.find(x)].append(x)
        return dict(groups)


# =============================================================================
# Coordinate utilities
# =============================================================================

def parse_tile_origin(filename: str) -> tuple:
    """
    Extract tile origin (x0, y0) from tile filename.

    Expected format: ..._x{x0}_y{y0}.jpg
    """
    match = re.search(r'_x(\d+)_y(\d+)\.jpg$', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    raise ValueError(f"Could not parse tile origin from filename: {filename}")


def local_to_global_bbox(local_xywh: list, tile_x0: int, tile_y0: int) -> list:
    """
    Convert tile-local bbox (xywh) to global bbox (xyxy).
    """
    x, y, w, h = local_xywh
    return [
        tile_x0 + x,
        tile_y0 + y,
        tile_x0 + x + w,
        tile_y0 + y + h
    ]


def compute_bbox_overlap(bbox1_xyxy: list, bbox2_xyxy: list) -> tuple:
    """
    Compute overlap ratios for two bboxes.

    Returns:
        (ratio1, ratio2) where ratio1 = intersection / area1, ratio2 = intersection / area2
    """
    x1_min, y1_min, x1_max, y1_max = bbox1_xyxy
    x2_min, y2_min, x2_max, y2_max = bbox2_xyxy

    # Compute intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
        return 0.0, 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    ratio1 = inter_area / area1 if area1 > 0 else 0.0
    ratio2 = inter_area / area2 if area2 > 0 else 0.0

    return ratio1, ratio2


def merge_bboxes(bboxes_xyxy: list) -> list:
    """
    Compute union bbox from list of bboxes (all in xyxy format).
    """
    if not bboxes_xyxy:
        return []

    x_min = min(b[0] for b in bboxes_xyxy)
    y_min = min(b[1] for b in bboxes_xyxy)
    x_max = max(b[2] for b in bboxes_xyxy)
    y_max = max(b[3] for b in bboxes_xyxy)

    return [x_min, y_min, x_max, y_max]


def xyxy_to_xywh(bbox_xyxy: list) -> list:
    """Convert xyxy to xywh format."""
    x_min, y_min, x_max, y_max = bbox_xyxy
    return [x_min, y_min, x_max - x_min, y_max - y_min]


# =============================================================================
# Main logic
# =============================================================================

def load_annotations(
    annotations_json_path: Path,
    tiles_dir: Path
) -> tuple:
    """
    Load tiled annotations and convert to global coordinates.

    Returns:
        (list of Annotation objects, category_id_to_name dict)
    """
    with open(annotations_json_path, 'r') as f:
        coco_data = json.load(f)

    # Build mappings
    image_id_to_file = {img['id']: img['file_name'] for img in coco_data['images']}
    category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}

    annotations = []
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        tile_file = image_id_to_file[image_id]

        # Extract just the filename (remove image_tiles/ prefix if present)
        tile_filename = Path(tile_file).name

        # Parse tile origin from filename
        tile_x0, tile_y0 = parse_tile_origin(tile_filename)

        # Convert local bbox to global
        local_bbox = ann['bbox']  # xywh
        global_bbox = local_to_global_bbox(local_bbox, tile_x0, tile_y0)

        annotations.append(Annotation(
            ann_id=ann['id'],
            image_id=image_id,
            tile_file=tile_filename,
            category_id=ann['category_id'],
            category_name=category_id_to_name.get(ann['category_id'], 'UNKNOWN'),
            local_bbox_xywh=local_bbox,
            global_bbox_xyxy=global_bbox,
            tile_x0=tile_x0,
            tile_y0=tile_y0
        ))

    return annotations, category_id_to_name


def find_duplicates(
    annotations: list,
    overlap_threshold: float = 0.8
) -> tuple:
    """
    Find duplicate annotations using bbox overlap.

    Returns:
        (UnionFind object with same-species duplicates, list of Conflict objects)
    """
    uf = UnionFind()
    conflicts = []
    duplicate_pairs = 0

    n = len(annotations)
    for i in range(n):
        for j in range(i + 1, n):
            ann_i = annotations[i]
            ann_j = annotations[j]

            ratio_i, ratio_j = compute_bbox_overlap(
                ann_i.global_bbox_xyxy,
                ann_j.global_bbox_xyxy
            )

            # Both ratios must exceed threshold
            if ratio_i >= overlap_threshold and ratio_j >= overlap_threshold:
                duplicate_pairs += 1

                # Check if species match
                if ann_i.category_id == ann_j.category_id:
                    # Same species: merge them
                    uf.union(ann_i.ann_id, ann_j.ann_id)
                else:
                    # Different species: conflict
                    conflicts.append(Conflict(
                        ann_id_1=ann_i.ann_id,
                        ann_id_2=ann_j.ann_id,
                        category_1=ann_i.category_name,
                        category_2=ann_j.category_name,
                        overlap_ratio_1=ratio_i,
                        overlap_ratio_2=ratio_j
                    ))

    return uf, conflicts, duplicate_pairs


def build_duplicate_groups(
    annotations: list,
    uf: UnionFind
) -> list:
    """
    Build DuplicateGroup objects from union-find clusters.
    """
    ann_id_to_ann = {a.ann_id: a for a in annotations}

    # Ensure all annotations are in union-find (singletons)
    for ann in annotations:
        uf.find(ann.ann_id)

    groups_dict = uf.get_groups()
    groups = []

    for group_id, (root, member_ids) in enumerate(groups_dict.items()):
        member_anns = [ann_id_to_ann[aid] for aid in member_ids]

        # All should have same category (we only merged same-species)
        category_id = member_anns[0].category_id
        category_name = member_anns[0].category_name

        # Compute merged bbox
        merged_bbox = merge_bboxes([a.global_bbox_xyxy for a in member_anns])

        groups.append(DuplicateGroup(
            group_id=group_id,
            annotation_ids=member_ids,
            annotations=member_anns,
            category_id=category_id,
            category_name=category_name,
            merged_bbox_xyxy=merged_bbox
        ))

    return groups


def crop_from_orthomosaic(
    img: 'np.ndarray',
    bbox_xyxy: list,
    padding: int = 0,
    min_size: int = 1
) -> Optional['np.ndarray']:
    """
    Crop region from image with optional padding.
    """
    img_h, img_w = img.shape[:2]

    x_min, y_min, x_max, y_max = [int(v) for v in bbox_xyxy]

    # Add padding
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(img_w, x_max + padding)
    y_max = min(img_h, y_max + padding)

    # Check minimum size
    if (x_max - x_min) < min_size or (y_max - y_min) < min_size:
        return None

    return img[y_min:y_max, x_min:x_max]


def generate_crop_filename(
    ortho_stem: str,
    index: int,
    species: str,
    bbox_xyxy: list
) -> str:
    """Generate parseable crop filename."""
    x_min, y_min, x_max, y_max = [int(v) for v in bbox_xyxy]
    w = x_max - x_min
    h = y_max - y_min
    return f"{ortho_stem}_bird_{index:06d}_{species}_x{x_min}_y{y_min}_w{w}_h{h}.jpg"


def deduplicate_and_crop(
    tiles_dir: Path,
    annotations_json_path: Path,
    output_root: Path,
    orthomosaic_path: Optional[Path] = None,
    overlap_threshold: float = 0.8,
    crop_padding: int = 0,
    min_crop_size: int = 10,
    jpeg_quality: int = 95
) -> dict:
    """
    Main deduplication and cropping pipeline.

    Args:
        tiles_dir: Directory containing tile images
        annotations_json_path: Path to tiled annotations.json
        output_root: Output directory root
        orthomosaic_path: Optional path to original orthomosaic for cropping
        overlap_threshold: Minimum overlap ratio for duplicate detection
        crop_padding: Pixels to add around crop bbox
        min_crop_size: Minimum crop dimension
        jpeg_quality: JPEG save quality

    Returns:
        Summary statistics dict
    """
    # Derive orthomosaic stem from annotations path
    ortho_stem = annotations_json_path.parent.name

    # Create output directories
    output_dir = output_root / ortho_stem
    crops_dir = output_dir / 'crops'
    metadata_dir = output_dir / 'metadata'
    crops_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Load annotations
    print(f"Loading annotations from {annotations_json_path}...")
    annotations, category_id_to_name = load_annotations(annotations_json_path, tiles_dir)
    print(f"  Loaded {len(annotations)} tiled annotations")

    # Find duplicates
    print(f"Finding duplicates (threshold={overlap_threshold})...")
    uf, conflicts, duplicate_pairs = find_duplicates(annotations, overlap_threshold)
    print(f"  Found {duplicate_pairs} duplicate candidate pairs")
    print(f"  Found {len(conflicts)} species conflicts")

    # Build groups
    groups = build_duplicate_groups(annotations, uf)
    merged_groups = [g for g in groups if len(g.annotation_ids) > 1]
    print(f"  Built {len(groups)} final groups ({len(merged_groups)} merged, {len(groups) - len(merged_groups)} singletons)")

    # Load image for cropping
    if orthomosaic_path and orthomosaic_path.exists():
        print(f"Loading orthomosaic from {orthomosaic_path}...")
        source_img = cv2.imread(str(orthomosaic_path))
        crop_source = 'orthomosaic'
    else:
        print("Warning: No orthomosaic provided, cropping from tiles not implemented")
        source_img = None
        crop_source = 'none'

    # Generate crops and metadata
    print("Generating crops...")
    deduplicated_records = []
    crops_saved = 0

    for idx, group in enumerate(groups):
        crop_filename = generate_crop_filename(
            ortho_stem,
            idx,
            group.category_name,
            group.merged_bbox_xyxy
        )

        # Crop from source image
        if source_img is not None:
            crop = crop_from_orthomosaic(
                source_img,
                group.merged_bbox_xyxy,
                padding=crop_padding,
                min_size=min_crop_size
            )
            if crop is not None:
                crop_path = crops_dir / crop_filename
                cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                crops_saved += 1

        # Build metadata record
        bbox_xywh = xyxy_to_xywh(group.merged_bbox_xyxy)
        record = {
            'crop_file': crop_filename,
            'category_id': group.category_id,
            'category_name': group.category_name,
            'merged_bbox_xyxy_global': [round(v, 2) for v in group.merged_bbox_xyxy],
            'merged_bbox_xywh_global': [round(v, 2) for v in bbox_xywh],
            'source_annotation_ids': group.annotation_ids,
            'source_tile_files': list(set(a.tile_file for a in group.annotations)),
            'duplicate_group_size': len(group.annotation_ids)
        }
        deduplicated_records.append(record)

    # Save metadata
    print("Saving metadata...")

    # deduplicated_annotations.json
    dedup_ann_path = metadata_dir / 'deduplicated_annotations.json'
    with open(dedup_ann_path, 'w') as f:
        json.dump({
            'orthomosaic': ortho_stem,
            'crop_source': crop_source,
            'total_crops': len(deduplicated_records),
            'categories': [{'id': cid, 'name': name} for cid, name in sorted(category_id_to_name.items())],
            'annotations': deduplicated_records
        }, f, indent=2)

    # duplicate_groups.json
    groups_data = []
    for g in groups:
        groups_data.append({
            'group_id': g.group_id,
            'category_id': g.category_id,
            'category_name': g.category_name,
            'merged_bbox_xyxy': [round(v, 2) for v in g.merged_bbox_xyxy],
            'member_count': len(g.annotation_ids),
            'members': [
                {
                    'ann_id': a.ann_id,
                    'tile_file': a.tile_file,
                    'global_bbox_xyxy': [round(v, 2) for v in a.global_bbox_xyxy]
                }
                for a in g.annotations
            ]
        })

    groups_path = metadata_dir / 'duplicate_groups.json'
    with open(groups_path, 'w') as f:
        json.dump({
            'orthomosaic': ortho_stem,
            'total_groups': len(groups),
            'merged_groups': len(merged_groups),
            'singleton_groups': len(groups) - len(merged_groups),
            'groups': groups_data
        }, f, indent=2)

    # conflicts.json
    conflicts_data = [
        {
            'ann_id_1': c.ann_id_1,
            'ann_id_2': c.ann_id_2,
            'category_1': c.category_1,
            'category_2': c.category_2,
            'overlap_ratio_1': round(c.overlap_ratio_1, 4),
            'overlap_ratio_2': round(c.overlap_ratio_2, 4)
        }
        for c in conflicts
    ]

    conflicts_path = metadata_dir / 'conflicts.json'
    with open(conflicts_path, 'w') as f:
        json.dump({
            'orthomosaic': ortho_stem,
            'total_conflicts': len(conflicts),
            'conflicts': conflicts_data
        }, f, indent=2)

    # Summary
    summary = {
        'orthomosaic': ortho_stem,
        'total_tiled_annotations': len(annotations),
        'duplicate_candidate_pairs': duplicate_pairs,
        'total_merged_groups': len(merged_groups),
        'total_conflicts': len(conflicts),
        'total_final_crops': len(deduplicated_records),
        'crops_saved': crops_saved,
        'output_dir': str(output_dir)
    }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Deduplicate bird detections and generate crops'
    )
    parser.add_argument(
        '--tiles-dir', '-t',
        type=Path,
        required=True,
        help='Directory containing tile images (image_tiles/)'
    )
    parser.add_argument(
        '--annotations', '-a',
        type=Path,
        required=True,
        help='Path to tiled annotations.json'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        required=True,
        help='Output root directory'
    )
    parser.add_argument(
        '--orthomosaic', '-m',
        type=Path,
        default=None,
        help='Path to original orthomosaic image (for cropping)'
    )
    parser.add_argument(
        '--overlap-threshold',
        type=float,
        default=0.8,
        help='Minimum overlap ratio for duplicate detection (default: 0.8)'
    )
    parser.add_argument(
        '--crop-padding',
        type=int,
        default=0,
        help='Pixels to pad around crop bbox (default: 0)'
    )
    parser.add_argument(
        '--min-crop-size',
        type=int,
        default=10,
        help='Minimum crop dimension in pixels (default: 10)'
    )
    parser.add_argument(
        '--jpeg-quality',
        type=int,
        default=95,
        help='JPEG save quality (default: 95)'
    )

    args = parser.parse_args()

    result = deduplicate_and_crop(
        tiles_dir=args.tiles_dir,
        annotations_json_path=args.annotations,
        output_root=args.output,
        orthomosaic_path=args.orthomosaic,
        overlap_threshold=args.overlap_threshold,
        crop_padding=args.crop_padding,
        min_crop_size=args.min_crop_size,
        jpeg_quality=args.jpeg_quality
    )

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Orthomosaic:              {result['orthomosaic']}")
    print(f"Total tiled annotations:  {result['total_tiled_annotations']}")
    print(f"Duplicate candidate pairs:{result['duplicate_candidate_pairs']}")
    print(f"Merged groups:            {result['total_merged_groups']}")
    print(f"Species conflicts:        {result['total_conflicts']}")
    print(f"Final crops:              {result['total_final_crops']}")
    print(f"Crops saved:              {result['crops_saved']}")
    print(f"Output:                   {result['output_dir']}")


if __name__ == '__main__':
    main()
