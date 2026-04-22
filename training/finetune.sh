#!/usr/bin/env bash
# Fine-tune GR00T N1.6 on any recorded dataset (SO-101 + IMX274).
#
# Usage:
#   bash training/finetune.sh <dataset_folder> <checkpoint_folder> [extra args...]
#
# Arguments:
#   dataset_folder     Name of the subfolder under datasets/  (e.g. control_blue_only)
#   checkpoint_folder  Name of the subfolder under checkpoints/ to save into
#
# Examples:
#   bash training/finetune.sh control_blue_only control_blue_only
#   bash training/finetune.sh experiment_mixed_8020 experiment_mixed_8020
#   bash training/finetune.sh control_blue_only control_blue_only --resume
#
# Any extra arguments after the two required ones are passed directly to
# launch_finetune.py (e.g. --max-steps 5000 --resume).

set -euo pipefail

# ── Argument validation ────────────────────────────────────────────────────────
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <dataset_folder> <checkpoint_folder> [extra args...]"
    echo ""
    echo "Examples:"
    echo "  $0 control_blue_only     control_blue_only"
    echo "  $0 experiment_mixed_8020 experiment_mixed_8020"
    echo "  $0 control_blue_only     control_blue_only --resume"
    exit 1
fi

DATASET_FOLDER="$1"
CHECKPOINT_FOLDER="$2"
shift 2   # remaining args ($@) are passed through to launch_finetune.py

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJ_DIR="$HOME/hsb-groot-robot"
GROOT_DIR="$HOME/Isaac-GR00T"
BASE_MODEL="$PROJ_DIR/models/base/GR00T-N1.6-3B"
DATASET="$PROJ_DIR/datasets/$DATASET_FOLDER"
CONFIG="$PROJ_DIR/training/so101_imx274_config.py"
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
echo "  Output     : $OUTPUT"
echo "──────────────────────────────────────────────────"

# ── Environment setup ──────────────────────────────────────────────────────────
cd "$GROOT_DIR"
source scripts/activate_thor.sh 2>/dev/null || true

VENV_CUDA_LIBS="$GROOT_DIR/.venv/lib/python3.12/site-packages/nvidia/cu13/lib"
export LD_LIBRARY_PATH="$VENV_CUDA_LIBS:${LD_LIBRARY_PATH:-}"

VENV_PYTHON="$GROOT_DIR/.venv/bin/python3"

# ── torchcodec: ensure the FFmpeg-6 build is installed ─────────────────────────
# uv sync / uv run will reinstall the Jetson PyPI wheel (built against FFmpeg 7,
# which is not on this system). We keep a source-built FFmpeg-6 wheel in
# Isaac-GR00T/wheels/ and reinstall it here if needed — takes ~3 s.
TC_WHEEL="$GROOT_DIR/wheels/torchcodec-0.10.0a0-cp312-cp312-linux_aarch64.whl"
TC_CORE6="$GROOT_DIR/.venv/lib/python3.12/site-packages/torchcodec/libtorchcodec_core6.so"
if [[ ! -f "$TC_CORE6" ]]; then
    echo "torchcodec: FFmpeg-6 build missing — reinstalling from cached wheel..."
    uv pip install --python "$VENV_PYTHON" --no-deps "$TC_WHEEL"
    echo "torchcodec: reinstalled OK"
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
    --global-batch-size 16 \
    --dataloader-num-workers 4 \
    "$@"
