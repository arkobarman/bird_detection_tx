# Chester Island Seabird Detection Dataset (2025)

## Overview

This dataset contains high-resolution orthomosaic imagery and pixel-level annotations of colonial waterbirds collected during the 2025 nesting season on **Chester Island, Texas**, a coastal nesting habitat monitored as part of long-running waterbird conservation efforts.

Colonial waterbirds serve as important indicators of ecosystem health, and their population trends are critical for informing environmental management and conservation policy. Traditional monitoring methods—such as on-foot surveys, boat-based observation, or manned aerial surveys—are labor-intensive, costly, and may disturb sensitive habitats.

To address these challenges, recent efforts have leveraged **unmanned aerial vehicles (UAVs)** to safely and efficiently collect high-resolution imagery of nesting colonies. However, manually counting and identifying birds from hundreds or thousands of aerial images remains a significant bottleneck.

This dataset is part of an ongoing collaboration between the **Houston Audubon Society** and the **Rice University D2K Lab**, aimed at developing automated pipelines for bird detection, counting, and species classification using deep learning.

The dataset consists of:

- **24 orthomosaic images (~10,000 × 10,000 pixels each)** covering Chester Island
- **Pixel-level segmentation masks, bounding boxes, and species labels** for each bird instance
- **16,002 annotated bird instances across 25 species categories**

Annotations were generated through a semi-automated process:

- Initial segmentation masks and bounding boxes were produced using a deep learning model (CO-DETR)
- Outputs were **manually reviewed and corrected by Hank Arnold**, including adding missed birds and removing false detections
- Species labels were assigned using expert knowledge and field survey data

This dataset is released to support research in:

- Aerial wildlife monitoring
- Object detection and instance segmentation
- Fine-grained species classification
- Scalable ecological survey automation

By combining UAV imagery with modern computer vision techniques, this work aims to enable **more accurate, scalable, and less intrusive monitoring of bird populations**, ultimately supporting conservation efforts and ecosystem health assessment.

## Imagery Collection and Processing

### Data Collection

Imagery is collected using a **DJI M300 RTK UAV** equipped with a **DJI P1 camera** fitted with a 35mm lens. The P1/35mm combination captures images at **8192 × 5460 pixels** at 72 dpi.

| Attribute | Value |
|-----------|-------|
| Platform | DJI M300 RTK UAV |
| Camera | DJI P1 with 35mm lens |
| Image resolution | 8192 × 5460 pixels at 72 dpi |
| Flight altitude | 35–40 meters (depending on battery constraints) |
| Flight pattern | Autonomous grid with ≥70% horizontal and vertical overlap |

Imagery collection missions are flown using an autonomous flight grid with at least 70% horizontal image overlap and 70% vertical image overlap. If battery requirements allow, slower speeds for increased vertical overlap and/or closer grid lines for increased horizontal overlap are used. The DJI P1 camera saves all images to an SD card, which is then used to transfer the raw results to a computer.

### Orthomosaic Generation

After copying the raw images to a computer, **Agisoft Metashape** is used to photogrammetrically create a single orthomosaic image of the entire survey area in full resolution.

Each individual bird could potentially appear in as many as **6 different raw images** due to the overlapping flight pattern. Photogrammetry is used to create a georeferenced view that contains each bird a single time.

### Tiling to 10k Images

The resulting orthomosaic image of the entire survey area would be too large for any further processing, so a set of **non-overlapping, full-resolution tile images** are exported. Each tile has an associated **JGW "World File"** defining the relationship between the individual pixels in that image to WGS Latitude and Longitude coordinates.

We currently export these images as **10,000 × 10,000 pixel JPG files**, although larger or smaller tiles could be used. These full-resolution, geo-referenced JPG images are referred to as **"10k files"**.

## Dataset Structure

```
BirdDataset_2025_10k/
├── images/                  # 24 orthomosaic images + 1 raw drone image
├── annotations/
│   ├── all_annotations.json       # Combined annotations for all images
│   ├── OM_001_annotations.json    # Individual annotations per image
│   ├── OM_002_annotations.json
│   └── ...
├── metadata/                # Summary statistics and species counts
└── README.md
```

- 24 orthomosaic images
- 1 raw drone image (reference)
- COCO-style JSON annotations including:
  - segmentation masks
  - bounding boxes
  - species labels

## Annotations

- **Total annotated bird instances:** 16,002
- **Total species categories:** 25

### Annotation File Formats

#### `all_annotations.json`

Combined annotations for all images in a single file:

```json
{
  "images": [
    {
      "file_name": "OM_002_20250510_10k-03-12.jpg",
      "width": 10000,
      "height": 10000,
      "detections": [
        {
          "bbox": [x_min, y_min, x_max, y_max],
          "score": 0.97,
          "category_id": 1,
          "category_name": "Bird",
          "tcws_species": "LAGUA",
          "segmentation": [[x1, y1, x2, y2, ...]]
        }
      ]
    }
  ]
}
```

#### Individual Image Annotations (e.g., `OM_001_annotations.json`)

Per-image annotation files:

```json
{
  "image": {
    "file_name": "OM_001_20250510_10k-01-14.jpg",
    "width": 10000,
    "height": 10000
  },
  "annotations": [
    {
      "bbox": [x_min, y_min, x_max, y_max],
      "score": 0.81,
      "category_id": 1,
      "category_name": "Bird",
      "tcws_species": "BRPEA",
      "segmentation": [[x1, y1, x2, y2, ...]]
    }
  ]
}
```

#### Field Descriptions

| Field | Description |
|-------|-------------|
| `bbox` | Bounding box as `[x_min, y_min, x_max, y_max]` in pixels |
| `score` | Detection confidence score (0-1) |
| `category_id` | Object category ID (1 = Bird) |
| `category_name` | Object category name |
| `tcws_species` | Texas Colonial Waterbird Survey species code |
| `segmentation` | Polygon mask as flattened `[x1, y1, x2, y2, ...]` coordinates |

### Species Counts

Code suffix convention: **A** = adult, **C** = chick, **F** = flying, **N** = nesting.

| Species Code | Common Name | Count |
|--------------|-------------|-------|
| ROTEA | Royal Tern (adult) | 8,392 |
| SATEA | Sandwich Tern (adult) | 3,043 |
| BRPEC | Brown Pelican (chick) | 1,963 |
| LAGUA | Laughing Gull (adult) | 1,234 |
| BRPEA | Brown Pelican (adult) | 579 |
| BLSKA | Black Skimmer (adult) | 147 |
| TRHEA | Tricolored Heron (adult) | 143 |
| GREGC | Great Egret (chick) | 99 |
| GREGA | Great Egret (adult) | 87 |
| RUTUA | Ruddy Turnstone (adult) | 63 |
| LWBBA | Large White Bird Above Canopy (adult) | 59 |
| GBHEC | Great Blue Heron (chick) | 47 |
| WHIBA | White Ibis (adult) | 40 |
| ROSPA | Roseate Spoonbill (adult) | 25 |
| OTHRA | Other / Non-nesting species (adult) | 23 |
| DCCOA | Double-crested Cormorant (adult) | 18 |
| GBHEA | Great Blue Heron (adult) | 17 |
| SNEGA | Snowy Egret (adult) | 5 |
| UNSURE | Unidentified | 5 |
| CATEA | Cattle Egret (adult) | 4 |
| AMAVA | American Avocet (adult) | 4 |
| ROSPC | Roseate Spoonbill (chick) | 2 |
| REEGWM | Reddish Egret (white morph) | 1 |
| BLSTA | Black-necked Stilt (adult) | 1 |
| REEGA | Reddish Egret (adult) | 1 |
| **Total** | | **16,002** |

## Annotation Process

Annotations were generated through a semi-automated process:

1. **Initial detection:** Segmentation masks and bounding boxes were produced using a pre-trained CO-DETR model
2. **Manual review and correction by Hank Arnold:**
   - Adding missed birds that the model failed to detect
   - Removing false detections
   - Correcting mask boundaries
3. **Species labeling:** Labels assigned based on expert knowledge and field survey data

Approximately 12,000+ masks were manually edited to ensure high-quality annotations.

## Tiling for Model Training

- Training images generated from 10k orthomosaics
- Typical tile size: 512 × 512 pixels
- Overlap used to avoid partial detections
- Each bird is guaranteed to appear in at least one tile

## Intended Use

This dataset is designed for:

- Bird detection (object detection / segmentation)
- Instance segmentation (mask prediction)
- Species classification
- Ecological population estimation
- Human-AI collaborative annotation systems

## Limitations

- Dataset is limited to 2025 collection only
- Species distribution is highly imbalanced (dominance of tern species)
- Dense nesting regions may still introduce:
  - Overlapping masks
  - Occlusions
- Some rare species have very few examples

## Future Work

Future versions of this dataset may include:

- Additional years (2023–2024)
- Expanded species coverage
- Improved consistency across annotation protocols

## Credits

Dataset collected and annotated by:

- **Hank Arnold** (Houston Audubon Society)
- **Rice University D2K Lab**
