#!/bin/bash
# Install dependencies for the GR00T robot control pipeline inside the
# Holoscan Docker container.  Run this once after entering the container:
#
#   bash install_robot_deps.sh
#
# The script expects that the host's home directory is mounted via
# docker's  -v $HOME:$HOME  flag (already set in run.sh).

set -euo pipefail

echo "=== Installing robot control dependencies ==="

# ---------- pip packages ----------

# pyzmq + msgpack — ZMQ transport & serialization for PolicyClient <-> PolicyServer
# pyserial        — serial port driver for Feetech servos
# deepdiff        — required by lerobot.motors internals
# tqdm            — progress bars used in lerobot calibration
# feetech-servo-sdk — provides the scservo_sdk driver
pip3 install --quiet pyzmq msgpack pyserial deepdiff tqdm feetech-servo-sdk
echo "[OK] pip packages installed"

# ---------- lerobot (editable, no-deps) ----------

LEROBOT_SRC="/home/latticeapp/lerobot"
if [ -d "$LEROBOT_SRC" ]; then
    pip3 install --quiet --no-deps -e "$LEROBOT_SRC"
    echo "[OK] lerobot installed (editable, no-deps) from $LEROBOT_SRC"
else
    echo "[WARN] lerobot source not found at $LEROBOT_SRC — install manually"
fi

# ---------- PYTHONPATH for gr00t ----------

GROOT_SRC="/home/latticeapp/Isaac-GR00T"
if [ -d "$GROOT_SRC" ]; then
    echo "[OK] Isaac-GR00T found at $GROOT_SRC"
else
    echo "[WARN] Isaac-GR00T not found at $GROOT_SRC — install manually"
fi

echo ""
echo "=== Done.  PYTHONPATH is set automatically by run.sh. ==="
echo ""
echo "Run the player:"
echo ""
echo "  # With robot control:"
echo "    python3 linux_imx274_player.py \\"
echo "        --robot-port /dev/ttyACM0 \\"
echo "        --robot-id my_awesome_follower_arm \\"
echo "        --policy-host localhost \\"
echo "        --policy-port 5555 \\"
echo "        --lang-instruction 'Move the blue dice'"
echo ""
echo "  # Camera only (no robot):"
echo "    python3 linux_imx274_player.py --no-robot"
