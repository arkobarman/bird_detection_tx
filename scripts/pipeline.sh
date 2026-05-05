#!/bin/bash
#
# End-to-end training data pipeline
#
# Usage:
#   ./pipeline.sh --images-dir DIR --annotations JSON [options]
#
# Required:
#   --images-dir   DIR   Directory of source orthophoto images (e.g. hank-updated-images/)
#   --annotations  JSON  Path to all-annotations JSON (e.g. hank-updated-json.json)
#
# Optional:
#   --samples-dir  DIR   Directory of sample/reference files to include in dataset
#   --tile-size    N     Tile size in pixels (default: 500)
#   --output-dir   DIR   Root output directory (default: <repo_root>/data)
#   --splits-dir   DIR   Root splits directory (default: <repo_root>/splits)
#
# Steps:
#   1. prepare_bird_dataset.py  → data/BirdDataset_2025_10k_MMDDYYYY/
#   2. run_tiling.sh            → data/BirdDataset_2025_nonoverlapping_tiles_<size>_MMDDYYYY/
#   3. run_splitting.sh         → splits/detection_tile_splits_MMDDYYYY/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Defaults ──────────────────────────────────────────────────────────────────
IMAGES_DIR=""
ANNOTATIONS_JSON=""
SAMPLES_DIR=""
TILE_SIZE=500
OUTPUT_DIR="${REPO_ROOT}/data"
SPLITS_DIR="${REPO_ROOT}/splits"

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --images-dir)   IMAGES_DIR="$2";       shift 2 ;;
        --annotations)  ANNOTATIONS_JSON="$2"; shift 2 ;;
        --samples-dir)  SAMPLES_DIR="$2";      shift 2 ;;
        --tile-size)    TILE_SIZE="$2";         shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";        shift 2 ;;
        --splits-dir)   SPLITS_DIR="$2";        shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "${IMAGES_DIR}" ] || [ -z "${ANNOTATIONS_JSON}" ]; then
    echo "Usage: $0 --images-dir DIR --annotations JSON [options]"
    echo ""
    echo "Required:"
    echo "  --images-dir   DIR   Source orthophoto images directory"
    echo "  --annotations  JSON  All-annotations JSON file"
    echo ""
    echo "Optional:"
    echo "  --samples-dir  DIR   Sample/reference files to include (default: none)"
    echo "  --tile-size    N     Tile size in pixels (default: 500)"
    echo "  --output-dir   DIR   Root output directory (default: <repo>/data)"
    echo "  --splits-dir   DIR   Root splits directory (default: <repo>/splits)"
    exit 1
fi

# ── Shared date stamp ─────────────────────────────────────────────────────────
DATE=$(date +%m%d%Y)

DATASET_DIR="${OUTPUT_DIR}/BirdDataset_2025_10k_${DATE}"
TILES_DIR="${OUTPUT_DIR}/BirdDataset_2025_nonoverlapping_tiles_${TILE_SIZE}_${DATE}"
SPLIT_OUT="${SPLITS_DIR}/detection_tile_splits_${DATE}"

# ── Print plan ────────────────────────────────────────────────────────────────
echo "============================================================"
echo "Bird Detection Training Pipeline"
echo "============================================================"
echo "Date stamp     : ${DATE}"
echo "Images dir     : ${IMAGES_DIR}"
echo "Annotations    : ${ANNOTATIONS_JSON}"
echo "Tile size      : ${TILE_SIZE}px"
echo ""
echo "Step 1 output  : ${DATASET_DIR}"
echo "Step 2 output  : ${TILES_DIR}"
echo "Step 3 output  : ${SPLIT_OUT}"
echo "============================================================"
echo ""

# ── Step 1: Prepare dataset ───────────────────────────────────────────────────
echo "[1/3] Preparing dataset..."

PREPARE_SCRIPT="${SCRIPT_DIR}/utils/prepare_bird_dataset.py"
PREPARE_ARGS=(
    "${IMAGES_DIR}"
    "${ANNOTATIONS_JSON}"
    --output-dir "${OUTPUT_DIR}"
)
[ -n "${SAMPLES_DIR}" ] && PREPARE_ARGS+=(--samples-dir "${SAMPLES_DIR}")

python3 "${PREPARE_SCRIPT}" "${PREPARE_ARGS[@]}"

echo ""
echo "[1/3] Done → ${DATASET_DIR}"
echo ""

# ── Step 2: Tile orthomosaics ─────────────────────────────────────────────────
echo "[2/3] Tiling orthomosaics..."

bash "${SCRIPT_DIR}/tiling/run_tiling.sh" \
    --dataset-dir "${DATASET_DIR}" \
    --output-dir  "${TILES_DIR}" \
    --tile-size   "${TILE_SIZE}"

echo ""
echo "[2/3] Done → ${TILES_DIR}"
echo ""

# ── Step 3: Generate splits ───────────────────────────────────────────────────
echo "[3/3] Generating train/val/test splits..."

bash "${SCRIPT_DIR}/splitting/run_splitting.sh" \
    --tiles-dir  "${TILES_DIR}" \
    --output-dir "${SPLIT_OUT}"

echo ""
echo "[3/3] Done → ${SPLIT_OUT}"
echo ""

# ── Summary ───────────────────────────────────────────────────────────────────
echo "============================================================"
echo "Pipeline complete"
echo "============================================================"
echo "Dataset  : ${DATASET_DIR}"
echo "Tiles    : ${TILES_DIR}"
echo "Splits   : ${SPLIT_OUT}"
echo "============================================================"
