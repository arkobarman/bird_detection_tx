#!/bin/bash
#
# Generate train/val/test splits for the tiled bird detection dataset
#
# Usage:
#   ./run_splitting.sh --tiles-dir DIR --output-dir DIR [--iterations N] [--seed N]
#
# Options:
#   --tiles-dir   DIR   Path to tiled dataset folder with all_annotations.json (required)
#   --output-dir  DIR   Path to write split files (required)
#   --iterations  N     Optimization iterations (default: 5000)
#   --seed        N     Random seed (default: 42)

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults
DATASET_DIR=""
OUTPUT_DIR=""
ITERATIONS=5000
SEED=42

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tiles-dir)   DATASET_DIR="$2"; shift 2 ;;
        --output-dir)  OUTPUT_DIR="$2";  shift 2 ;;
        --iterations)  ITERATIONS="$2";  shift 2 ;;
        --seed)        SEED="$2";        shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "${DATASET_DIR}" ] || [ -z "${OUTPUT_DIR}" ]; then
    echo "Usage: $0 --tiles-dir DIR --output-dir DIR [--iterations N] [--seed N]"
    exit 1
fi

ANNOTATIONS="${DATASET_DIR}/all_annotations.json"

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
