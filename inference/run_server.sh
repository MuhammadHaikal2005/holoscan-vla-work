#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Start the GR00T N1.6 policy inference server for the dual-USB-camera SO-101.
#
# Run this in Terminal 1, then run eval_dual_usb.py in Terminal 2.
#
# Usage:
#   bash inference/run_server.sh [checkpoint_folder]
#
# Default checkpoint: checkpoints/dual_cam_blue_only/checkpoint-2000
# ──────────────────────────────────────────────────────────────────────────────

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GROOT_DIR="$HOME/Isaac-GR00T"
VENV_PYTHON="$GROOT_DIR/.venv/bin/python3"

# ── Checkpoint ─────────────────────────────────────────────────────────────────
CHECKPOINT_FOLDER="${1:-checkpoints/dual_cam_blue_only/checkpoint-2000}"
CHECKPOINT="$REPO_ROOT/$CHECKPOINT_FOLDER"

if [[ ! -d "$CHECKPOINT" ]]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT"
    exit 1
fi

# ── Environment ────────────────────────────────────────────────────────────────
# activate_thor.sh uses find/pipe internally which can return non-zero;
# source it without strict error mode to avoid false early exits.
source "$GROOT_DIR/scripts/activate_thor.sh" || true

# activate_thor.sh only scans the system site-packages for nvidia/*/lib.
# The venv has its own copies (cudnn, cudss, etc.) — prepend those too.
VENV_NVIDIA_LIBS=$(find "$GROOT_DIR/.venv/lib/python3.12/site-packages/nvidia" -name "lib" -type d 2>/dev/null | tr '\n' ':')
if [[ -n "$VENV_NVIDIA_LIBS" ]]; then
    export LD_LIBRARY_PATH="${VENV_NVIDIA_LIBS}${LD_LIBRARY_PATH:-}"
fi

echo ""
echo "──────────────────────────────────────────────────────"
echo "  GR00T Policy Server"
echo "  Checkpoint : $CHECKPOINT"
echo "  Port       : 5555"
echo "──────────────────────────────────────────────────────"
echo ""

cd "$GROOT_DIR"
"$VENV_PYTHON" gr00t/eval/run_gr00t_server.py \
    --model_path "$CHECKPOINT" \
    --embodiment_tag NEW_EMBODIMENT \
    --port 5555
