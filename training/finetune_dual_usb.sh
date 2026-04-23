#!/usr/bin/env bash
# Fine-tune GR00T N1.6 on a dual-USB-camera dataset (SO-101 + cam0 + cam1).
#
# Usage:
#   bash training/finetune_dual_usb.sh <dataset_folder> <checkpoint_folder> [extra args...]
#
# Arguments:
#   dataset_folder     Name of the subfolder under datasets/  (e.g. dual_cam_blue_only)
#   checkpoint_folder  Name of the subfolder under checkpoints/ to save into
#
# Examples:
#   bash training/finetune_dual_usb.sh dual_cam_blue_only dual_cam_blue_only
#   bash training/finetune_dual_usb.sh dual_cam_mixed_8020 dual_cam_mixed_8020
#   bash training/finetune_dual_usb.sh dual_cam_blue_only dual_cam_blue_only --resume
#
# Any extra arguments after the two required ones are passed directly to
# launch_finetune.py (e.g. --max-steps 5000 --resume).

set -euo pipefail

# ── Argument validation ────────────────────────────────────────────────────────
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <dataset_folder> <checkpoint_folder> [extra args...]"
    echo ""
    echo "Examples:"
    echo "  $0 dual_cam_blue_only   dual_cam_blue_only"
    echo "  $0 dual_cam_mixed_8020  dual_cam_mixed_8020"
    echo "  $0 dual_cam_blue_only   dual_cam_blue_only --resume"
    exit 1
fi

DATASET_FOLDER="$1"
CHECKPOINT_FOLDER="$2"
shift 2

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJ_DIR="$HOME/hsb-groot-robot"
GROOT_DIR="$HOME/Isaac-GR00T"
BASE_MODEL="$PROJ_DIR/models/base/GR00T-N1.6-3B"
DATASET="$PROJ_DIR/datasets/$DATASET_FOLDER"
CONFIG="$PROJ_DIR/training/so101_dual_usb_config.py"   # dual-camera modality config
OUTPUT="$PROJ_DIR/checkpoints/$CHECKPOINT_FOLDER"

# ── Sanity checks ──────────────────────────────────────────────────────────────
if [[ ! -d "$DATASET" ]]; then
    echo "ERROR: Dataset not found: $DATASET"
    exit 1
fi

if [[ ! -f "$DATASET/meta/modality.json" ]]; then
    echo "ERROR: Missing modality.json in $DATASET/meta/ — run setup first."
    exit 1
fi

mkdir -p "$OUTPUT"

echo "──────────────────────────────────────────────────"
echo "  Base model : $BASE_MODEL"
echo "  Dataset    : $DATASET"
echo "  Config     : $CONFIG"
echo "  Output     : $OUTPUT"
echo "──────────────────────────────────────────────────"

# ── Environment setup ──────────────────────────────────────────────────────────
cd "$GROOT_DIR"
source scripts/activate_thor.sh 2>/dev/null || true

VENV_CUDA_LIBS="$GROOT_DIR/.venv/lib/python3.12/site-packages/nvidia/cu13/lib"
export LD_LIBRARY_PATH="$VENV_CUDA_LIBS:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

VENV_PYTHON="$GROOT_DIR/.venv/bin/python3"

# ── torchcodec: verify the x86_64 wheel loaded correctly ──────────────────────
if ! "$VENV_PYTHON" -c "import torchcodec" 2>/dev/null; then
    echo "ERROR: torchcodec failed to load. Try: sudo apt install ffmpeg"
    exit 1
fi

# ── Launch ─────────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 "$VENV_PYTHON" gr00t/experiment/launch_finetune.py \
    --base-model-path "$BASE_MODEL" \
    --dataset-path    "$DATASET" \
    --embodiment-tag  NEW_EMBODIMENT \
    --modality-config-path "$CONFIG" \
    --num-gpus 1 \
    --output-dir      "$OUTPUT" \
    --max-steps       2000 \
    --save-steps      500 \
    --global-batch-size 8 \
    --dataloader-num-workers 2 \
    "$@"
