# Seabird Detection & Classification Pipeline

This repository contains the data processing pipelines and tooling for building a seabird detection and classification system from high-resolution aerial orthomosaics.

Part of a research collaboration between the **Houston Audubon Society** and the **Rice University D2K Lab**.

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FULL PROCESSING PIPELINE                          │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  1. RAW DATA     │     │  2. TILING       │     │  3. SPLITTING    │
│                  │     │                  │     │                  │
│  BirdDataset_    │ ──▶ │  run_tiling.sh   │ ──▶ │ run_splitting.sh │
│  2025_10k/       │     │                  │     │                  │
│  (10k×10k ortho) │     │  → 500×500 tiles │     │  → train/val/test│
└──────────────────┘     └──────────────────┘     └──────────────────┘
                                                           │
                                                           ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  6. CLASSIFICA-  │     │  5. CROPPING     │     │  4. DETECTION    │
│     TION         │     │                  │     │                  │
│                  │ ◀── │ crop_coco_tiles  │ ◀── │  CO-DETR model   │
│  Swin/CNN model  │     │     .py          │     │  (external)      │
│  on crops        │     │  → bird crops    │     │  → COCO bbox     │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

---

## Step-by-Step Pipeline

### Step 1: Raw Orthomosaic Data

Place the raw orthomosaic dataset in `data/BirdDataset_2025_10k/`:

```
data/BirdDataset_2025_10k/
├── images/
│   ├── OM_001_20250510_10k-03-11.jpg    # 10,000×10,000 pixel orthomosaics
│   ├── OM_002_20250510_10k-03-12.jpg
│   └── ...
├── annotations/
│   ├── OM_001_annotations.json           # Per-orthomosaic annotations
│   ├── OM_002_annotations.json
│   └── ...
├── metadata/
│   └── species_totals.csv                # Species distribution summary
└── README.md
```

**Source:** Internal Google Drive / Dropbox (see `data/README.md` for links)

---

### Step 2: Tiling (Orthomosaic → 500×500 Tiles)

Run the tiling script to generate fixed-size tiles from orthomosaics:

```bash
./scripts/tiling/run_tiling.sh
```

**Input:** `data/BirdDataset_2025_10k/`
**Output:** `data/BirdDataset_2025_nonoverlapping_tiles_500/`

```
data/BirdDataset_2025_nonoverlapping_tiles_500/
├── OM_001/
│   └── tiles/
│       ├── OM_001_00000_00000.png        # Tile at position (0, 0)
│       ├── OM_001_00000_00500.png        # Tile at position (0, 500)
│       └── ...
├── OM_002/
│   └── tiles/
│       └── ...
├── all_annotations.json                   # Combined COCO annotations for all tiles
└── README.md
```

**Note:** Birds on tile boundaries may appear in multiple tiles (~16% annotation inflation).

---

### Step 3: Splitting (Train/Val/Test for Detection)

Generate optimized train/val/test splits for detection model training:

```bash
./scripts/splitting/run_splitting.sh
```

**Input:** `data/BirdDataset_2025_nonoverlapping_tiles_500/all_annotations.json`
**Output:** `splits/detection_tile_splits/`

```
splits/detection_tile_splits/
├── train.json          # 80% of tiles
├── val.json            # 10% of tiles
├── test.json           # 10% of tiles
├── split_report.txt    # Species distribution per split
└── split_summary.json
```

The splitting algorithm optimizes for balanced species distribution across splits.

---

### Step 4: Detection (CO-DETR)

*External step - run separately*

Train a CO-DETR detection model on the tiled dataset using the splits from Step 3. The detection model produces COCO-style bounding box annotations.

**Input:** Tiles + `splits/detection_tile_splits/`
**Output:** Updated `all_annotations.json` with detection results

---

### Step 5: Cropping (Detections → Bird Crops)

Crop individual birds from tiles using bounding box annotations:

```bash
python scripts/utils/crop_coco_tiles.py \
    --json data/BirdDataset_2025_nonoverlapping_tiles_500/all_annotations.json \
    --tile-dir data/BirdDataset_2025_nonoverlapping_tiles_500 \
    --output-dir data/cropped-dataset
```

**Input:** Tiles + COCO annotations
**Output:** `data/cropped-dataset/`

```
data/cropped-dataset/
├── crops/
│   ├── ROTEA/
│   │   ├── ROTEA_000001.png              # 224×224 cropped bird images
│   │   ├── ROTEA_000002.png
│   │   └── ...
│   ├── SATEA/
│   │   └── ...
│   └── {other species}/
├── annotations.csv                        # Crop metadata for classification
└── species_distribution.json              # Species counts summary
```

---

### Step 6: Classification

*Separate training step*

Train a classification model (Swin Transformer, CNN) on the cropped bird images.

**Input:** `data/cropped-dataset/`
**Output:** Trained classification model

---

## Repository Structure

```
bird_detection_tx/
├── data/
│   ├── BirdDataset_2025_10k/                    # Raw orthomosaics (Step 1)
│   ├── BirdDataset_2025_nonoverlapping_tiles_500/  # Tiled dataset (Step 2)
│   └── cropped-dataset/                         # Bird crops (Step 5)
│
├── splits/
│   └── detection_tile_splits/                   # Train/val/test splits (Step 3)
│
├── scripts/
│   ├── tiling/
│   │   ├── run_tiling.sh                        # Step 2 entry point
│   │   └── tile_orthomosaics_nonoverlapping.py
│   ├── splitting/
│   │   ├── run_splitting.sh                     # Step 3 entry point
│   │   └── find_best_splits.py
│   ├── utils/
│   │   └── crop_coco_tiles.py                   # Step 5 cropping script
│   └── visualization/
│       ├── plot_species_distribution.py         # Original dataset EDA
│       ├── plot_tiled_distribution.py           # Tiled dataset EDA
│       └── plot_om_distributions.py             # Per-orthomosaic EDA
│
├── figures/
│   └── data_exploration/
│       ├── original_dataset_distribution.png
│       ├── tiled_dataset_distribution.png
│       └── om_distributions/
│
└── README.md
```

---

## Quick Start

```bash
# 1. Clone repository
git clone <repo_url>
cd bird_detection_tx

# 2. Download dataset (see data/README.md for links)
# Place BirdDataset_2025_10k/ in data/

# 3. Run tiling
./scripts/tiling/run_tiling.sh

# 4. Generate splits
./scripts/splitting/run_splitting.sh

# 5. (Train detection model externally with CO-DETR)

# 6. Crop detections for classification
python scripts/utils/crop_coco_tiles.py \
    --json data/BirdDataset_2025_nonoverlapping_tiles_500/all_annotations.json \
    --tile-dir data/BirdDataset_2025_nonoverlapping_tiles_500 \
    --output-dir data/cropped-dataset
```

---

## Data Format

### COCO-Style Annotations

```json
{
  "info": {"description": "...", "tile_size": 500},
  "categories": [
    {"id": 1, "name": "ROTEA"},
    {"id": 2, "name": "SATEA"}
  ],
  "images": [
    {"id": 1, "file_name": "OM_001/tiles/OM_001_00000_00000.png", "width": 500, "height": 500}
  ],
  "annotations": [
    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [x, y, width, height]}
  ]
}
```

### Species Codes

| Code | Species |
|------|---------|
| ROTEA | Royal Tern Adults |
| SATEA | Sandwich Tern Adults |
| BRPEC | Brown Pelican Chicks |
| BRPEA | Brown Pelican Adults |
| LAGUA | Laughing Gull Adults |
| TRHEA | Tri-Colored Heron Adults |
| ... | (see species_totals.csv for full list) |

---

## EDA & Visualization

Generate dataset distribution plots:

```bash
# Original 10k orthomosaic distribution
python scripts/visualization/plot_species_distribution.py

# Tiled 500×500 distribution
python scripts/visualization/plot_tiled_distribution.py

# Per-orthomosaic distributions
python scripts/visualization/plot_om_distributions.py
```

---

## Credits

- **Hank Arnold** (Houston Audubon Society) - Data collection & annotation
- **Rice University D2K Lab** - Pipeline development & model training
