# Chester Island Seabird Detection Dataset - Tiled (2025)

## Overview

This dataset contains **500×500 pixel tiles** derived from high-resolution orthomosaic imagery of colonial waterbirds, collected during the 2025 nesting season on **Chester Island, Texas**. The tiles are designed for training object detection and instance segmentation models.

This is the **tiled version** of the Chester Island Seabird Detection Dataset (2025). The original dataset contains full orthomosaic images (~10,000 × 10,000 pixels), while this version provides model-ready tiles with COCO-format annotations.

### Dataset Statistics

| Attribute             | Value            |
| --------------------- | ---------------- |
| Source orthomosaics   | 22               |
| Total tiles           | 8,800            |
| Tiles per orthomosaic | 400 (20×20 grid) |
| Tile size             | 500 × 500 pixels |
| Total annotations     | 17,479           |
| Species categories    | 35               |

## Dataset Structure

```
BirdDataset_2025_nonoverlapping_tiles_500/
├── all_annotations.json          # Combined COCO annotations for all tiles
├── OM_001/
│   ├── tiles/
│   │   ├── OM_001_00000_00000.png
│   │   ├── OM_001_00000_00500.png
│   │   ├── OM_001_00500_00000.png
│   │   └── ...
│   ├── annotations.json          # Per-orthomosaic COCO annotations
│   └── tiling_diagnostics.json   # Tiling statistics
├── OM_002/
│   └── ...
└── OM_022/
    └── ...
```

### Tile Naming Convention

Tiles are named using the format: `OM_{id}_{x:05d}_{y:05d}.png`

- `OM_{id}`: Orthomosaic identifier (e.g., OM_001)
- `{x:05d}`: X-coordinate of the **top-left corner** of the tile (zero-padded to 5 digits)
- `{y:05d}`: Y-coordinate of the **top-left corner** of the tile (zero-padded to 5 digits)

Example: `OM_001_00500_01000.png` is a tile from orthomosaic OM_001 with its top-left corner at pixel position (500, 1000) in the source orthomosaic.

## Annotations

Annotations are provided in **COCO format**.

### Combined Annotations (`all_annotations.json`)

```json
{
  "info": {
    "description": "Seabird Detection Dataset - All Orthomosaics Tiled",
    "version": "1.0",
    "tile_size": 500,
    "num_orthomosaics": 22
  },
  "categories": [
    {"id": 1, "name": "AMAVA"},
    {"id": 2, "name": "BNSTA"},
    ...
  ],
  "images": [
    {
      "id": 1,
      "file_name": "OM_001/tiles/OM_001_00000_00000.png",
      "width": 500,
      "height": 500,
      "orthomosaic_id": "OM_001",
      "tile_x": 0,
      "tile_y": 0
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 5,
      "category_id": 14,
      "bbox": [x, y, width, height],
      "area": 4774.0,
      "iscrowd": 0
    }
  ]
}
```

### Annotation Fields

| Field         | Description                                                       |
| ------------- | ----------------------------------------------------------------- |
| `bbox`        | Bounding box as `[x, y, width, height]` in tile-local coordinates |
| `area`        | Bounding box area in pixels                                       |
| `category_id` | Species category ID (see categories list)                         |
| `iscrowd`     | Always 0 (no crowd annotations)                                   |
| `image_id`    | Reference to the tile image                                       |

### Coordinate System

- **Origin**: Top-left corner of each tile
- **Bbox format**: `[x, y, width, height]` (COCO standard)
- **Units**: Pixels (tile-local coordinates)

## Species Distribution

| Code    | Count      | Species Name                                      |
| ------- | ---------- | ------------------------------------------------- |
| ROTEA   | 9,100      | Royal Tern Adults                                 |
| SATEA   | 2,751      | Sandwich Tern Adults                              |
| BRPEC   | 2,490      | Brown Pelican Chicks                              |
| LAGUA   | 1,433      | Laughing Gull Adults                              |
| BRPEA   | 819        | Brown Pelican Adults                              |
| TRHEA   | 172        | Tri-Colored Heron Adults                          |
| GREGC   | 107        | Great Egret Chicks                                |
| GREGA   | 106        | Great Egret Adults                                |
| LWBBA   | 69         | Large White Bird Above Canopy                     |
| GBHEC   | 60         | Great Blue Heron Chicks                           |
| WHIBA   | 45         | White Ibis Adults                                 |
| ROTEF   | 44         | Royal Tern Flying                                 |
| MTRNS   | 44         | Mixed Terns (Nesting/Sitting)                     |
| RUTUA   | 42         | Ruddy Turnstone Adults                            |
| ROSPA   | 32         | Roseate Spoonbill Adults                          |
| OTHRA   | 27         | Other (Not nesting species)                       |
| GBHEA   | 25         | Great Blue Heron Adults                           |
| DCCOA   | 20         | Double-Crested Cormorant Adults                   |
| UNSURE  | 19         | Unsure/Unidentified                               |
| BRPEF   | 15         | Brown Pelican Flying                              |
| LAGUF   | 11         | Laughing Gull Flying                              |
| SATEF   | 11         | Sandwich Tern Flying                              |
| SNEGA   | 11         | Snowy Egret Adults                                |
| CATEA   | 5          | Caspian Tern Adults                               |
| AMAVA   | 5          | American Avocet Adults                            |
| DCCOF   | 4          | Double-Crested Cormorant Flying                   |
| WHIBF   | 2          | White Ibis Flying                                 |
| GREGF   | 2          | Great Egret Flying                                |
| ROSPC   | 2          | Roseate Spoonbill Chicks                          |
| REEGWMA | 1          | Reddish Egret White Morph Adults                  |
| BNSTA   | 1          | Black-Necked Stilt Adults                         |
| REEGA   | 1          | Reddish Egret Adults                              |
| NECOA   | 1          | Neotropic Cormorant Adults                        |
| RUTUF   | 1          | Ruddy Turnstone Flying                            |
| REEGF   | 1          | Reddish Egret Flying                              |
| **Total** | **17,479** |                                                 |

## Tiling Process

Tiles were generated from 10,000×10,000 pixel orthomosaic images using the following process:

1. **Grid-based tiling**: Non-overlapping 500×500 pixel tiles extracted from each orthomosaic
2. **Edge handling**: Incomplete edge tiles are dropped (10,000 ÷ 500 = 20 tiles per dimension, no remainder)
3. **Annotation mapping**: Bounding boxes converted from orthomosaic coordinates to tile-local coordinates
4. **Partial boxes**: Birds crossing tile boundaries are clipped and included in both adjacent tiles

### Tiling Statistics (per orthomosaic)

| Metric                       | Value |
| ---------------------------- | ----- |
| Tiles per orthomosaic        | 400   |
| Dropped edge tiles           | 0     |
| Average annotations per tile | ~2.0  |

## Intended Use

This dataset is designed for:

- **Object detection** training and evaluation
- **Instance segmentation** (when combined with mask annotations)
- **Species classification** fine-tuning
- **Ecological population estimation** research

## Relationship to Source Dataset

| Attribute    | Source (10k)    | Tiled (500) |
| ------------ | --------------- | ----------- |
| Image size   | 10,000 × 10,000 | 500 × 500   |
| Total images | 22              | 8,800       |
| Annotations  | 15,015          | 17,479*     |
| Format       | Custom JSON     | COCO        |

*Annotation count is higher in the tiled version because birds crossing tile boundaries appear in multiple tiles.

## Limitations

- No overlapping tiles (birds at boundaries may be partially visible)
- Species distribution is highly imbalanced
- Some tiles contain no birds (empty tiles)
- Partial/clipped bounding boxes at tile edges

## Credits

Dataset collected and annotated by:

- **Hank Arnold** (Houston Audubon Society)
- **Rice University D2K Lab**

Tiling pipeline developed as part of the bird detection research project.
