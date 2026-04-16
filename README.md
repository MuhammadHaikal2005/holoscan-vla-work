# HSB + GR00T Robot Control

Runs the NVIDIA Holoscan Sensor Bridge (HSB) IMX274 camera through a GPU
pipeline and feeds frames to a GR00T N1.6 policy server to control an SO-101
robot arm.

---

## How it works

```
IMX274 camera (ethernet, 192.168.0.2)
  └─ LinuxReceiverOp       reassembles UDP packets into one raw frame
      └─ CsiToBayerOp      raw CSI  →  Bayer uint16  (GPU)
          └─ ImageProcessorOp  optical black subtraction, white balance
              └─ BayerDemosaicOp  Bayer  →  RGBA uint16  (GPU)
                  ├─ HolovizOp "holoviz"
                  │       hardware sRGB curve at scanout  →  display window
                  │       (always on)
                  │
                  ├─ SrgbConvertOp          [only with --preview]
                  │       drops alpha, applies sRGB in CuPy, emits RGB uint8
                  │       stays 100% on GPU — no CPU copy here
                  │   ├─ HolovizOp "preview"
                  │   │       shows exactly what GR00T receives
                  │   └─ PolicyClientOp     [only with robot mode]
                  │           skips its own conversion (uint8 detected)
                  │
                  └─ PolicyClientOp         [robot mode, no --preview]
                          applies sRGB inline, reads joint state from SO-101,
                          sends frame + state to GR00T server over ZMQ,
                          receives 16-step action chunk
                      └─ RobotActionOp
                              writes Goal_Position to SO-101 servos via serial
                              optionally logs predicted / actual positions to CSV
```

---

## Setup (one-time)

### 1 — Start the GR00T policy server (host, outside Docker)

```bash
cd ~/Isaac-GR00T
LD_LIBRARY_PATH=/home/latticeapp/Isaac-GR00T/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-} \
uv run python -m gr00t.eval.run_gr00t_server \
    --model_path ~/groot-so101-finetune/model --port 5555
```

Wait for `Server listening on port 5555` before running the player.

> Uses `uv run` (not conda) because only the `Isaac-GR00T/.venv` has the
> correct pinned GR00T deps and the CUDA DSS library.

### 2 — Launch the Docker container (host)

```bash
xhost +          # allow X11 from Docker (once per session)
cd ~/hsb-groot-robot
./run.sh
```

`run.sh` mounts `$HOME`, `/dev`, the X11 socket, and sets `PYTHONPATH` so
`import holoscan` and `import gr00t` work inside the container.

### 3 — Install robot dependencies (inside container, first run only)

```bash
bash install_robot_deps.sh
```

Installs `pyzmq`, `msgpack`, `pyserial`, `feetech-servo-sdk`, `lerobot`
(editable), `opencv-python-headless`, and `pyfakewebcam`.

---

## Running linux_imx274_player.py

### Camera only — no robot

```bash
python3 linux_imx274_player.py --camera-mode 1 --no-robot
```

### Camera + preview window (see exactly what GR00T receives)

```bash
python3 linux_imx274_player.py --camera-mode 1 --no-robot --preview --exposure 0.3
```

Opens two windows: the raw demosaiced feed and the sRGB uint8 frame the model sees.

### Full pipeline — camera + GR00T + robot arm

```bash
python3 linux_imx274_player.py \
    --camera-mode 1 \
    --policy-host localhost \
    --policy-port 5555 \
    --lang-instruction "Move the blue dice" \
    --exposure 0.3
```

### Full pipeline with preview + CSV logging

```bash
python3 linux_imx274_player.py \
    --camera-mode 1 \
    --lang-instruction "Move the blue dice" \
    --exposure 0.3 \
    --preview \
    --action-log /home/latticeapp/actions.csv \
    --sent-log /home/latticeapp/sent_commands.csv
```

CSV files land in `$HOME` which is bind-mounted, so they persist after the container exits.

---

## Arguments — linux_imx274_player.py

### Camera

| Argument | Default | Description |
|---|---|---|
| `--camera-mode` | `0` | Camera resolution mode. `0` = 4K, `1` = 1080p, `2` = 4K alt. |
| `--hololink` | `192.168.0.2` | IP address of the HSB board. |
| `--expander-configuration` | `0` | I2C expander config (`0` or `1`). |
| `--pattern` | off | Display a built-in test pattern (0–11) instead of live camera. |

### Display

| Argument | Default | Description |
|---|---|---|
| `--headless` | off | Run without any display window. |
| `--fullscreen` | off | Open the Holoviz window in fullscreen. |
| `--preview` | off | Open a second window showing the exact RGB uint8 frame sent to GR00T. Runs `SrgbConvertOp` on GPU — no extra CPU copy. |
| `--exposure` | `0.3` | Linear brightness multiplier applied before the sRGB curve. Raise to brighten, lower to darken. Applied in linear light space so the sRGB tone curve stays correct. |

### Robot control

| Argument | Default | Description |
|---|---|---|
| `--no-robot` | off | Skip all robot and policy code. Camera and display only. |
| `--robot-port` | `/dev/ttyACM0` | Serial port for the SO-101 follower arm. |
| `--robot-id` | `my_awesome_follower_arm` | Calibration ID — must match the filename in `~/.cache/huggingface/lerobot/calibration/robots/so_follower/`. |
| `--policy-host` | `localhost` | Hostname where the GR00T policy server is running. |
| `--policy-port` | `5555` | ZMQ port of the GR00T policy server. |
| `--action-horizon` | `8` | Steps to execute from each inference chunk before requesting a new one. Lower = more responsive, higher = smoother. |
| `--lang-instruction` | `"Move the blue dice"` | Natural language task description sent to the model each inference call. |

### Logging

| Argument | Default | Description |
|---|---|---|
| `--action-log` | off | CSV file path. Logs predicted joint goal positions before each `sync_write`. |
| `--sent-log` | off | CSV file path. Logs sent commands plus present positions read back after each `sync_write` — lets you compare commanded vs actual. |

### Other

| Argument | Default | Description |
|---|---|---|
| `--frame-limit` | off | Exit after N frames. Useful for quick tests. |
| `--configuration` | `example_configuration.yaml` | Path to the Holoscan YAML config. |
| `--log-level` | `20` (INFO) | Python logging level. `10` = DEBUG, `20` = INFO, `30` = WARNING. |

---

## Running linux_imx274_player_v4l2_sink.py

This script is a **comparison tool**. It runs the same camera pipeline but
writes frames to a v4l2loopback device and/or a named pipe so you can visually
compare the sRGB and old linear conversions. There is no robot control here.

### Setup (host, before running)

```bash
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label=HolovizBridge
```

### Typical usage

```bash
# sRGB output (default) → /dev/video10
python3 linux_imx274_player_v4l2_sink.py --camera-mode 1 --v4l2-device /dev/video10

# Old linear output — for before/after comparison
python3 linux_imx274_player_v4l2_sink.py --camera-mode 1 --v4l2-device /dev/video10 --linear

# View the loopback feed on the host
ffplay -f v4l2 -input_format yuyv422 -video_size 1920x1080 /dev/video10
```

### View raw RGB uint8 via named pipe (no v4l2loopback needed)

```bash
# Inside container:
python3 linux_imx274_player_v4l2_sink.py \
    --camera-mode 1 --no-v4l2-sink \
    --rgb-pipe /home/latticeapp/rgb_feed

# Host (separate terminal):
ffplay -f rawvideo -pixel_format rgb24 -video_size 1920x1080 /home/latticeapp/rgb_feed
```

### Arguments — linux_imx274_player_v4l2_sink.py

| Argument | Default | Description |
|---|---|---|
| `--camera-mode` | `0` | Same as main player. |
| `--v4l2-device` | `/dev/video10` | Path to the v4l2loopback device. |
| `--no-v4l2-sink` | off | Skip the v4l2 sink. Use with `--rgb-pipe` or to run Holoviz only. |
| `--linear` | off | Use the old `255/4095 * 0.7` linear scale instead of sRGB. |
| `--exposure` | `0.3` | Same as main player. |
| `--rgb-pipe` | off | Path to a named pipe. Writes raw RGB24 frames so `ffplay` can display them on the host without v4l2loopback. |
| `--headless` | off | No display window. |
| `--fullscreen` | off | Fullscreen Holoviz window. |
| `--frame-limit` | off | Exit after N frames. |
| `--hololink` | `192.168.0.2` | HSB board IP. |
| `--log-level` | `20` | Logging level. |

---

## Camera modes

| `--camera-mode` | Resolution | Notes |
|---|---|---|
| `0` | 3840×2160 (4K) | Default |
| `1` | 1920×1080 (1080p) | Recommended for robot control |
| `2` | 3840×2160 (4K alt) | |

---

## Files

| File | Purpose |
|---|---|
| `linux_imx274_player.py` | Main app — camera + optional sRGB preview + GR00T + robot arm |
| `linux_imx274_player_v4l2_sink.py` | Comparison tool — sRGB vs linear, named pipe RGB viewer |
| `example_configuration.yaml` | Holoscan YAML config used by both scripts |
| `run.sh` | Docker launcher — mounts, PYTHONPATH, env vars |
| `install_robot_deps.sh` | One-time pip install of robot deps inside the container |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'holoscan'`**
Exit the container and re-run `./run.sh`. The script sets `PYTHONPATH` automatically.

**`ModuleNotFoundError: No module named 'lerobot'` / `'cv2'` / `'pyfakewebcam'`**
Run `bash install_robot_deps.sh` inside the container.

**`ModuleNotFoundError: No module named 'gr00t'`**
`run.sh` adds `~/Isaac-GR00T` to `PYTHONPATH`. Verify that directory exists on the host.

**`ERROR: v4l2loopback device not found: /dev/video10`**
Create the device on the **host** first:
`sudo modprobe v4l2loopback devices=1 video_nr=10 card_label=HolovizBridge`

**`Failed to initialize glfw` / X11 errors**
Run `xhost +` on the host before launching the container.

**`FeetechMotorsBus: Missing motor IDs`**
Check the arm is powered on, USB cable connected, and the port is correct (`ls /dev/ttyACM*`).
If permission denied: `chmod 666 /dev/ttyACM0` inside the container.

**`ImportError: libcudss.so.0`** (policy server)
Use the full `LD_LIBRARY_PATH` prefix shown in Setup step 1. Do not use conda envs.

**Image too bright or too dark**
Adjust `--exposure`. Default is `0.3`. The value multiplies the linear sensor
data before the sRGB curve, so colour relationships stay correct.
