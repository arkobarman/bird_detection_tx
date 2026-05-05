#!/bin/bash
#
# Tile all orthomosaic images into NxN patches
#
# Usage:
#   ./run_tiling.sh --dataset-dir DIR --output-dir DIR [--tile-size N]
#
# Options:
#   --dataset-dir DIR   Path to BirdDataset_2025_10k_* folder (required)
#   --output-dir  DIR   Path to write tiled output (required)
#   --tile-size   N     Tile size in pixels (default: 500)

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get the repository root (two levels up from scripts/tiling/)
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Defaults
TILE_SIZE=500
DATASET_DIR=""
OUTPUT_BASE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset-dir) DATASET_DIR="$2"; shift 2 ;;
        --output-dir)  OUTPUT_BASE="$2";  shift 2 ;;
        --tile-size)   TILE_SIZE="$2";    shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "${DATASET_DIR}" ] || [ -z "${OUTPUT_BASE}" ]; then
    echo "Usage: $0 --dataset-dir DIR --output-dir DIR [--tile-size N]"
    exit 1
fi

SOURCE_IMAGES="${DATASET_DIR}/images"
SOURCE_ANNOTATIONS="${DATASET_DIR}/annotations"
TILING_SCRIPT="${SCRIPT_DIR}/tile_orthomosaics_nonoverlapping.py"

# Verify dataset exists
if [ ! -d "${DATASET_DIR}" ]; then
    echo "ERROR: Dataset not found at ${DATASET_DIR}"
    exit 1
fi

echo "=============================================="
echo "Orthomosaic Tiling Pipeline"
echo "=============================================="
echo "Tile size: ${TILE_SIZE}x${TILE_SIZE}"
echo "Source images: ${SOURCE_IMAGES}"
echo "Source annotations: ${SOURCE_ANNOTATIONS}"
echo "Output base: ${OUTPUT_BASE}"
echo "=============================================="
echo ""

# Create output base directory
mkdir -p "${OUTPUT_BASE}"

# Count images
IMAGE_COUNT=$(ls -1 "${SOURCE_IMAGES}"/OM_*.jpg 2>/dev/null | wc -l | tr -d ' ')
echo "Found ${IMAGE_COUNT} orthomosaic images to process"
echo ""

# Process each orthomosaic
PROCESSED=0
FAILED=0

for IMAGE_PATH in "${SOURCE_IMAGES}"/OM_*.jpg; do
    # Extract filename
    FILENAME=$(basename "${IMAGE_PATH}")

    # Extract OM_XXX ID (e.g., OM_001 from OM_001_20250510_10k-03-11.jpg)
    OM_ID=$(echo "${FILENAME}" | grep -oE 'OM_[0-9]{3}')

    if [ -z "${OM_ID}" ]; then
        echo "WARNING: Could not extract OM ID from ${FILENAME}, skipping"
        ((FAILED++))
        continue
    fi

    # Find corresponding annotation file
    ANNOTATION_PATH="${SOURCE_ANNOTATIONS}/${OM_ID}_annotations.json"

    if [ ! -f "${ANNOTATION_PATH}" ]; then
        echo "WARNING: Annotation file not found: ${ANNOTATION_PATH}, skipping"
        ((FAILED++))
        continue
    fi

    # Output directory for this orthomosaic
    OUTPUT_DIR="${OUTPUT_BASE}/${OM_ID}"

    echo "Processing ${OM_ID}..."
    echo "  Image: ${FILENAME}"
    echo "  Annotations: ${OM_ID}_annotations.json"
    echo "  Output: ${OUTPUT_DIR}"

    # Run tiling script
    python3 "${TILING_SCRIPT}" \
        -i "${IMAGE_PATH}" \
        -a "${ANNOTATION_PATH}" \
        -o "${OUTPUT_DIR}" \
        -t "${TILE_SIZE}"

    ((PROCESSED++))
    echo ""
done

echo "=============================================="
echo "INDIVIDUAL TILING COMPLETE"
echo "=============================================="
echo "Processed: ${PROCESSED} orthomosaics"
echo "Failed: ${FAILED} orthomosaics"
echo ""

# Merge all annotations into one combined file
echo "Merging annotations..."
MERGE_SCRIPT="${SCRIPT_DIR}/../utils/merge_annotations.py"
COMBINED_ANNOTATIONS="${OUTPUT_BASE}/all_annotations.json"

python3 "${MERGE_SCRIPT}" \
    -i "${OUTPUT_BASE}" \
    -o "${COMBINED_ANNOTATIONS}" \
    -t "${TILE_SIZE}"

echo ""
echo "=============================================="
echo "ALL COMPLETE"
echo "=============================================="
echo "Output: ${OUTPUT_BASE}"
echo "Combined annotations: ${COMBINED_ANNOTATIONS}"
echo "=============================================="
