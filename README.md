# HSB + GR00T Robot Control

Minimal repo for running the NVIDIA Holoscan Sensor Bridge (HSB) IMX274 camera
with GR00T N1 policy-driven SO-101 robot arm control on Jetson Thor.

## Prerequisites

| Dependency | Location | Notes |
|---|---|---|
| Docker image | `hololink-demo:2.5.0` | Pre-built on this machine |
| Isaac-GR00T | `~/Isaac-GR00T` | PolicyClient + server |
| lerobot | `~/lerobot` | Motor bus drivers |
| Model weights | `~/groot-so101-finetune/model` | Fine-tuned GR00T N1.6 |
| HSB board | `192.168.0.2` (default) | Connected via `enP2p1s0` |
| SO-101 arm | `/dev/ttyACM0` (default) | Feetech SCS servos |

## Quick start

```bash
# 1. Allow X11 forwarding (on the host, once per session)
xhost +

# 2. Enter the repo directory
cd ~/hsb-groot-robot

# 3. Launch the Docker container (interactive shell)
./run.sh

# 4. First time only: install robot dependencies
bash install_robot_deps.sh

# 5. Start the GR00T policy server (in a separate host terminal)
cd ~/Isaac-GR00T
conda activate lerobot2
python -m gr00t.eval.run_gr00t_server \
    --model_path ~/groot-so101-finetune/model --port 5555
```

## Running the player

### Camera + robot control (full pipeline)

```bash
python3 linux_imx274_player.py \
    --camera-mode 1 \
    --robot-port /dev/ttyACM0 \
    --policy-host localhost \
    --policy-port 5555 \
    --lang-instruction "Move the blue dice"
```

### Camera only (no robot)

```bash
python3 linux_imx274_player.py --camera-mode 1 --no-robot
```

### With action logging

```bash
python3 linux_imx274_player.py \
    --camera-mode 1 \
    --robot-port /dev/ttyACM0 \
    --policy-host localhost \
    --action-log /home/latticeapp/actions.csv \
    --sent-log /home/latticeapp/sent_commands.csv \
    --lang-instruction "Move the blue dice"
```

### V4L2 comparison build

To compare the Holoviz display against the V4L2 loopback output:

```bash
# Host: create the loopback device
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label=HolovizBridge

# Container:
python3 linux_imx274_player_v4l2_sink.py --camera-mode 1 --v4l2-device /dev/video10

# Host (separate terminal): view the fake camera
ffplay -f v4l2 -input_format yuyv422 -video_size 1920x1080 /dev/video10
```

## Files

| File | Purpose |
|---|---|
| `linux_imx274_player.py` | Main Holoscan app: camera pipeline + PolicyClientOp + RobotActionOp |
| `linux_imx274_player_v4l2_sink.py` | Legacy V4L2 loopback comparison build |
| `example_configuration.yaml` | Holoscan YAML config (referenced by both scripts) |
| `run.sh` | Docker launcher with all required mounts and env vars |
| `install_robot_deps.sh` | One-time pip install for robot dependencies inside the container |

## Architecture

```
HSB (IMX274)
  -> Ethernet packets (192.168.0.2)
    -> LinuxReceiverOp (reassembles frame)
      -> CsiToBayerOp (raw CSI -> Bayer uint16)
        -> ImageProcessorOp (optical black subtraction)
          -> BayerDemosaicOp (Bayer -> RGBA uint16)
            -> HolovizOp (display)
            -> PolicyClientOp (RGBA uint16 -> RGB uint8 -> GR00T server -> action chunk)
              -> RobotActionOp (sync_write Goal_Position to SO-101 servos)
```
