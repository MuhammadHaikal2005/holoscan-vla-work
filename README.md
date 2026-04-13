# HSB + GR00T Robot Control

Minimal repo for running the NVIDIA Holoscan Sensor Bridge (HSB) IMX274 camera
with GR00T N1 policy-driven SO-101 robot arm control on Jetson Thor.

## Prerequisites

| Dependency | Location | Notes |
|---|---|---|
| Docker image | `hololink-demo:2.5.0` | Pre-built on this machine |
| Isaac-GR00T | `~/Isaac-GR00T` | GR00T policy server + client |
| lerobot | `~/lerobot` | Motor bus drivers for SO-101 |
| Model weights | `~/groot-so101-finetune/model` | Fine-tuned GR00T N1.6 |
| HSB board | `192.168.0.2` (default) | Connected via `enP2p1s0` |
| SO-101 arm | `/dev/ttyACM0` (default) | Feetech SCS servos |

---

## Step 1 — Start the GR00T policy server (host terminal, outside Docker)

The server runs on the **host**, not inside Docker.

```bash
cd ~/Isaac-GR00T
LD_LIBRARY_PATH=/home/latticeapp/Isaac-GR00T/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-} \
uv run python -m gr00t.eval.run_gr00t_server \
    --model_path ~/groot-so101-finetune/model --port 5555
```

Wait until you see `Server listening on port 5555` before proceeding.

> **Why this command?**
> The `uv` venv inside `Isaac-GR00T/` has the correct pinned versions of all GR00T
> dependencies. The `LD_LIBRARY_PATH` line exposes `libcudss.so.0` which is bundled
> inside the venv but not on the default system path.
> Do NOT use `conda activate lerobot` or `conda activate lerobot2` —
> `lerobot` has a too-new `transformers`, and `lerobot2` has CPU-only PyTorch.

---

## Step 2 — Launch the Docker container (host terminal)

```bash
# Allow X11 forwarding (once per session)
xhost +

cd ~/hsb-groot-robot
./run.sh
```

This opens an interactive shell inside `hololink-demo:2.5.0`. The `run.sh` script:
- Mounts `$HOME`, `/dev`, X11 socket, and all required `/sys` paths
- Sets `PYTHONPATH` to include `/opt/nvidia/holoscan/python/lib` (required for `import holoscan`)
  and `/home/latticeapp/Isaac-GR00T` (required for `import gr00t`)

---

## Step 3 — Install robot dependencies (inside container, first time only)

```bash
bash install_robot_deps.sh
```

This installs `pyzmq`, `msgpack`, `pyserial`, `deepdiff`, `tqdm`, `feetech-servo-sdk`,
and the `lerobot` package (editable, no-deps) from `~/lerobot`.

---

## Step 4 — Run the player (inside container)

### Camera only — no robot (good for testing the pipeline works)

```bash
python3 linux_imx274_player.py --camera-mode 1 --no-robot
```

### Full pipeline — camera + GR00T + robot arm

```bash
python3 linux_imx274_player.py \
    --camera-mode 1 \
    --robot-port /dev/ttyACM0 \
    --policy-host localhost \
    --policy-port 5555 \
    --lang-instruction "Move the blue dice"
```

### With CSV action logging

```bash
python3 linux_imx274_player.py \
    --camera-mode 1 \
    --robot-port /dev/ttyACM0 \
    --policy-host localhost \
    --action-log /home/latticeapp/actions.csv \
    --sent-log /home/latticeapp/sent_commands.csv \
    --lang-instruction "Move the blue dice"
```

Logs are written to `$HOME` which is bind-mounted, so they persist after the container exits.

---

## V4L2 comparison build (optional)

To visually compare Holoviz output vs the fake V4L2 camera output:

```bash
# Host: create the loopback device
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label=HolovizBridge

# Inside container:
python3 linux_imx274_player_v4l2_sink.py --camera-mode 1 --v4l2-device /dev/video10

# Host (separate terminal):
ffplay -f v4l2 -input_format yuyv422 -video_size 1920x1080 /dev/video10
```

---

## Camera modes

| `--camera-mode` | Resolution | ffplay `-video_size` |
|---|---|---|
| `0` | 3840x2160 (4K) | `3840x2160` |
| `1` | 1920x1080 (1080p) | `1920x1080` |
| `2` | 3840x2160 (4K alt) | `3840x2160` |

---

## Architecture

```
HSB IMX274 camera
  --> Ethernet packets (192.168.0.2)
    --> LinuxReceiverOp       (reassembles UDP packets into one frame)
      --> CsiToBayerOp        (raw CSI -> Bayer uint16 on GPU)
        --> ImageProcessorOp  (optical black subtraction)
          --> BayerDemosaicOp (Bayer -> RGBA uint16 on GPU)
            --> HolovizOp     (display window)
            --> PolicyClientOp
                  reads joint state from SO-101 via serial
                  converts RGBA uint16 -> RGB uint8
                  sends frame + state to GR00T server (ZMQ port 5555)
                  receives 16-step action chunk
              --> RobotActionOp
                    writes Goal_Position to SO-101 servos via serial
                    (optional) logs predicted + actual positions to CSV
```

---

## Files

| File | Purpose |
|---|---|
| `linux_imx274_player.py` | Main Holoscan app: camera + PolicyClientOp + RobotActionOp |
| `linux_imx274_player_v4l2_sink.py` | Legacy V4L2 loopback comparison build |
| `example_configuration.yaml` | Holoscan YAML config (used by both scripts via `--configuration`) |
| `run.sh` | Docker launcher — sets PYTHONPATH and all required mounts |
| `install_robot_deps.sh` | One-time pip install of robot dependencies inside the container |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'holoscan'`**
The container was launched without the correct PYTHONPATH. Exit and re-run `./run.sh`
(the updated script sets `PYTHONPATH=/opt/nvidia/holoscan/python/lib:...` automatically).

**`ModuleNotFoundError: No module named 'lerobot'`**
Run `bash install_robot_deps.sh` inside the container first.

**`ModuleNotFoundError: No module named 'gr00t'`**
The `PYTHONPATH` in `run.sh` already includes `~/Isaac-GR00T`. If this still fails,
verify that `/home/latticeapp/Isaac-GR00T` exists on the host.

**`error: XDG_RUNTIME_DIR is invalid` / `Failed to initialize glfw`**
Run `xhost +` on the **host** before launching the container.

**`FeetechMotorsBus motor check failed — Missing motor IDs`**
- Check the arm is powered on and the USB cable is connected
- Confirm the port: `ls /dev/ttyACM*` inside the container
- If permission denied: `chmod 666 /dev/ttyACM0` inside the container

**`ImportError: libcudss.so.0` when starting the policy server**
Use the `LD_LIBRARY_PATH` prefix shown in Step 1. Do not use conda envs for the server.

**Policy server crashes with `ModuleNotFoundError: No module named 'tree'`**
You are using the wrong conda env. Use `uv run` as shown in Step 1.
