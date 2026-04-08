#!/bin/bash
#
# Generate train/val/test splits for the tiled bird detection dataset
#
# Usage:
#   ./run_splitting.sh
#
# This script generates 80/10/10 splits optimized for species distribution.

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get the repository root (two levels up from scripts/splitting/)
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Configuration
DATASET_DIR="${REPO_ROOT}/data/BirdDataset_2025_nonoverlapping_tiles_500"
ANNOTATIONS="${DATASET_DIR}/all_annotations.json"
OUTPUT_DIR="${REPO_ROOT}/splits/detection_tile_splits"
ITERATIONS=5000
SEED=42

# Verify input exists
if [ ! -f "${ANNOTATIONS}" ]; then
    echo "ERROR: Annotations file not found at ${ANNOTATIONS}"
    echo ""
    echo "Please ensure the tiled dataset exists with all_annotations.json"
    exit 1
fi

# Run split generation
python3 "${SCRIPT_DIR}/find_best_splits.py" \
    --tiles_dir "${DATASET_DIR}" \
    --annotations "${ANNOTATIONS}" \
    --output "${OUTPUT_DIR}" \
    --iterations "${ITERATIONS}" \
    --seed "${SEED}"
