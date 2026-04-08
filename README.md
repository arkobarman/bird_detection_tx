# Seabird Detection & Classification Dataset Pipeline

This repository contains the data processing pipelines, experiments, and tooling for building a large-scale seabird detection and classification pipeline from high-resolution aerial orthomosaics.

The project is part of a research collaboration with the D2K Lab at Rice University, focused on enabling automated wildlife monitoring using computer vision.

## Pipeline Overview

The full pipeline consists of three main stages:

### 1. Tiling & Annotation Generation

- Convert large orthomosaics into fixed-size tiles (e.g., 512x512)
- Use a **non-overlapping grid**
- Generate COCO-style bounding box annotations
- Preserve **partial birds at tile boundaries**
- Tile naming format:

  ```
  OM_{id}_{x}_{y}.png
  ```

  where `(x, y)` are zero-padded top-left coordinates

---

### 2. Deduplication & Cropping (Classification Dataset)

- Merge overlapping detections across tiles
- Remove duplicate birds caused by tiling overlap
- Crop individual birds into classification samples
- Produce a clean dataset for species classification

---

### 3. Model Training

- Detection:
  - Co-DETR trained on tiled dataset
- Classification:
  - Swin Transformer / CNN models trained on cropped birds

---

## Repository Structure

```
.
├── data/
│   ├── raw/                # Original orthomosaics + annotations
│   ├── tiles/              # Generated tiles
│   ├── annotations/        # COCO-style annotation files
│   └── crops/              # Cropped bird images (classification)
│
├── src/
│   ├── tiling/             # Tiling + annotation generation
│   ├── deduplication/      # Duplicate removal logic
│   ├── cropping/           # Bird crop extraction
│   └── utils/              # Shared utilities
│
├── experiments/
│   ├── detection/          # Co-DETR experiments
│   └── classification/     # Species classification experiments
│
├── scripts/
│   ├── run_tiling.py
│   ├── run_deduplication.py
│   └── run_training.py
│
├── outputs/
│   ├── logs/
│   └── reports/
│
└── README.md
```

---

## Data Format

We use **COCO-style annotations**:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "OM_003_00512_01024.png",
      "width": 512,
      "height": 512
    }
  ],
  "annotations": [
    {
      "image_id": 1,
      "bbox": [x, y, width, height],
      "category_id": 3,
      "area": 1024,
      "iscrowd": 0
    }
  ]
}
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone <repo_url>
cd bird_detection_tx
```

### 2. Set up environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Prepare the dataset

Place the `BirdDataset_2025_10k` folder in the `data/` directory:

```
data/BirdDataset_2025_10k/
├── images/
│   └── OM_*.jpg
└── annotations/
    └── OM_*_annotations.json
```

### 4. Run tiling pipeline

```bash
./scripts/tiling/run_tiling.sh
```

Or run manually:

```bash
python scripts/tiling/tile_orthomosaics_nonoverlapping.py \
    --input_dir data/raw \
    --output_dir data/tiles \
    --tile_size 512
```
