#!/bin/bash
# Install dependencies for the GR00T robot control pipeline inside the
# Holoscan Docker container.  Run this once after entering the container:
#
#   bash install_robot_deps.sh
#
# The script expects that the host's home directory is mounted via
# docker's  -v $HOME:$HOME  flag (already set in run.sh).

set -uo pipefail

echo "=== Installing robot control dependencies ==="

# ---------- pip packages ----------

# Packages are installed into /home/latticeapp/.docker-packages so they
# survive container restarts (that directory is on the mounted host volume).
# run.sh adds it to PYTHONPATH automatically.
DOCKER_PACKAGES="/home/latticeapp/.docker-packages"
mkdir -p "$DOCKER_PACKAGES"

# pyzmq              ZMQ transport for ZMQ camera publisher and PolicyClient
# msgpack            serialization for PolicyClient
# pyserial           serial port driver for Feetech servos
# deepdiff           required by lerobot.motors internals
# tqdm               progress bars
# feetech-servo-sdk  scservo_sdk driver
# opencv-python-headless  JPEG encoding in ZmqPublisherOp (no GUI needed)
pip3 install --quiet --root-user-action=ignore \
    --target "$DOCKER_PACKAGES" \
    pyzmq msgpack pyserial deepdiff tqdm feetech-servo-sdk opencv-python-headless
echo "[OK] pip packages installed to $DOCKER_PACKAGES"

# ---------- lerobot (editable, no-deps) ----------

LEROBOT_SRC="/home/latticeapp/lerobot"
if [ -d "$LEROBOT_SRC" ]; then
    pip3 install --quiet --no-deps --root-user-action=ignore -e "$LEROBOT_SRC"
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
echo "Run the inference pipeline:"
echo ""
echo "  # Camera + GR00T + robot arm:"
echo "    python pipeline/linux_imx274_player.py \\"
echo "        --camera-mode 1 \\"
echo "        --robot-port /dev/ttyACM0 \\"
echo "        --robot-id my_awesome_follower_arm \\"
echo "        --policy-host localhost \\"
echo "        --policy-port 5555 \\"
echo "        --lang-instruction 'Pick up the cube'"
echo ""
echo "  # Camera only (no robot):"
echo "    python pipeline/linux_imx274_player.py --camera-mode 1 --no-robot"
echo ""
echo "Run the ZMQ camera server (for dataset recording):"
echo ""
echo "    python pipeline/imx274_zmq_server.py --camera-mode 1 --headless"
