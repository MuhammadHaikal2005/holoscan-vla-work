#!/bin/bash
# Launch the hololink-demo Docker container with all required mounts
# for the HSB + GR00T robot pipeline.
#
# Usage:
#   ./run.sh                          # interactive shell
#   ./run.sh python3 linux_imx274_player.py --camera-mode 1 --no-robot
#
# Prerequisites (run once on the host):
#   xhost +

set -o errexit
set -o xtrace

IMAGE="hololink-demo:2.5.0"

# If no arguments, drop into a bash shell
if [ $# -eq 0 ]; then
    CMD="bash"
else
    CMD="$*"
fi

docker run \
    -it \
    --rm \
    --net host \
    --gpus all \
    --runtime=nvidia \
    --shm-size=1gb \
    --privileged \
    -v "$PWD":"$PWD" \
    -v "$HOME":"$HOME" \
    -v /sys/bus/pci/devices:/sys/bus/pci/devices \
    -v /sys/kernel/mm/hugepages:/sys/kernel/mm/hugepages \
    -v /dev:/dev \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /tmp/argus_socket:/tmp/argus_socket \
    -v /sys/devices:/sys/devices \
    -v /var/nvidia/nvcam/settings:/var/nvidia/nvcam/settings \
    -w "$PWD" \
    -e NVIDIA_DRIVER_CAPABILITIES=graphics,video,compute,utility,display \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e DISPLAY="$DISPLAY" \
    -e PYTHONPATH=/opt/nvidia/holoscan/python/lib:/home/latticeapp/Isaac-GR00T:${PYTHONPATH:-} \
    -e enableRawReprocess=2 \
    "$IMAGE" \
    $CMD
