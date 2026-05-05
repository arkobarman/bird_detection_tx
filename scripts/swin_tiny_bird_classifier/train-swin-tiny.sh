#!/bin/bash
#
# Train Swin-Tiny bird classifier
#
# Usage:
#   ./scripts/swin_tiny_bird_classifier/train-swin-tiny.sh [options]
#
# Options:
#   --split-dir    DIR   Path to split directory containing label_mapping.json (default: splits/swin-tiny-classifier1)
#   --image-root   DIR   Path to crops directory (default: data/BirdDataset_2025_05052026_crops/crops)
#   --epochs       N     Number of epochs (default: 30)
#   --lr           F     Learning rate (default: 1e-4)
#   --out-dir      DIR   Output directory for run (default: auto)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Defaults
SPLIT_DIR="${REPO_ROOT}/splits/swin-tiny-classifier1"
IMAGE_ROOT="${REPO_ROOT}/data/BirdDataset_2025_05052026_crops/crops"
EPOCHS=30
LR="1e-4"
OUT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --split-dir)   SPLIT_DIR="$2";  shift 2 ;;
        --image-root)  IMAGE_ROOT="$2"; shift 2 ;;
        --epochs)      EPOCHS="$2";     shift 2 ;;
        --lr)          LR="$2";         shift 2 ;;
        --out-dir)     OUT_DIR="$2";    shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

LABEL_MAPPING="${SPLIT_DIR}/label_mapping.json"

if [ ! -f "${LABEL_MAPPING}" ]; then
    echo "ERROR: label_mapping.json not found at ${LABEL_MAPPING}"
    exit 1
fi

if [ ! -d "${IMAGE_ROOT}" ]; then
    echo "ERROR: image root not found at ${IMAGE_ROOT}"
    exit 1
fi

echo "=============================================="
echo "Swin-Tiny Training"
echo "=============================================="
echo "Label mapping: ${LABEL_MAPPING}"
echo "Image root:    ${IMAGE_ROOT}"
echo "Epochs:        ${EPOCHS}"
echo "LR:            ${LR}"
echo "=============================================="
echo ""

cd "${REPO_ROOT}"

EXTRA_ARGS=""
if [ -n "${OUT_DIR}" ]; then
    EXTRA_ARGS="--out-dir ${OUT_DIR}"
fi

python3 -m scripts.swin_tiny_bird_classifier.SwinTrainer2025 \
    --label-mapping "${LABEL_MAPPING}" \
    --image-root    "${IMAGE_ROOT}" \
    --epochs        "${EPOCHS}" \
    --lr            "${LR}" \
    ${EXTRA_ARGS}
