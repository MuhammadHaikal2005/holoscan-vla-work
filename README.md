# HSB + GR00T Robot Control

Runs the NVIDIA Holoscan Sensor Bridge (HSB) IMX274 camera through a GPU
pipeline, feeds frames to a GR00T N1.6 policy server to control an SO-101
robot arm, and records LeRobot-compatible datasets for training.

---

## Repository layout

```
hsb-groot-robot/
├── run.sh                     Docker launcher (mounts, env vars, PYTHONPATH)
├── install_robot_deps.sh      One-time pip install inside the container
├── README.md
│
├── pipeline/                  Holoscan camera pipeline scripts (run inside Docker)
│   ├── linux_imx274_player.py       Main inference + teleop pipeline
│   ├── imx274_zmq_server.py         ZMQ camera publisher (for dataset recording)
│   ├── linux_imx274_player_v4l2_sink.py  Legacy v4l2 bridge (reference only)
│   └── example_configuration.yaml  Holoscan YAML config
│
├── recording/                 Dataset recording scripts (run on host)
│   └── imx274_lerobot_record.py     LeRobot dataset recorder (ZMQ camera input)
│
├── payload/                   HSB payload transmission experiments
│   ├── thor_send_payload.py
│   ├── thor_send_payload_with_ecb.py
│   └── test_udp_listener.py
│
├── payload_generator_op/      Custom C++ Holoscan operator (GPU tensor emitter)
│   ├── CMakeLists.txt
│   ├── build.sh
│   ├── payload_generator_op.{hpp,cpp}
│   └── payload_generator_op_pybind.cpp
│
├── references/                Upstream reference scripts
│   └── lerobot_record.py
│
├── notes/                     Debugging documentation
│   └── thor_send_payload_debugging.md
│
└── logs/                      Runtime log output (gitignored)
```

---

## How the inference pipeline works

```
IMX274 camera (ethernet, 192.168.0.2)
  └─ LinuxReceiverOp        reassembles UDP packets into one raw frame
      └─ CsiToBayerOp       raw CSI  →  Bayer uint16  (GPU)
          └─ ImageProcessorOp   optical black subtraction, white balance
              └─ BayerDemosaicOp    Bayer  →  RGBA uint16  (GPU)
                  ├─ HolovizOp "holoviz"
                  │       hardware sRGB curve at scanout  →  display window
                  │       (always on)
                  │
                  ├─ SrgbConvertOp          [only with --preview]
                  │       drops alpha, applies sRGB in CuPy, emits RGB uint8
                  │       stays 100% on GPU — no CPU copy
                  │   ├─ HolovizOp "preview"
                  │   │       shows exactly what GR00T receives
                  │   └─ PolicyClientOp     [robot mode + --preview]
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

## How the dataset recording pipeline works

```
Docker container                         Host
────────────────                         ────
imx274_zmq_server.py                     imx274_lerobot_record.py
  LinuxReceiverOp                          ZMQCamera
    CsiToBayerOp                             ↑ JPEG over ZMQ (port 5556)
      ImageProcessorOp             FeetechMotorsBus (follower, USB)
        BayerDemosaicOp            FeetechMotorsBus (leader,   USB) ← optional
          SrgbConvertOp                LeRobotDataset.add_frame()
            ZmqPublisherOp ──────→       └─ Parquet + video (LeRobot v2)
```

No v4l2loopback, no YUYV422 conversion, no lag — the RGB uint8 CuPy
tensor from `SrgbConvertOp` is JPEG-encoded on the GPU and sent directly
over ZMQ to the recording script on the host.

---

## Setup (one-time)

### 1 — Start the GR00T policy server (host, outside Docker)

> Only needed for inference (`linux_imx274_player.py`), not for recording.

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
(editable), and `opencv-python-headless`.

---

## Inference — linux_imx274_player.py

> Run inside Docker via `./run.sh python pipeline/linux_imx274_player.py ...`

### Camera only — no robot

```bash
python pipeline/linux_imx274_player.py --camera-mode 1 --no-robot
```

### Camera + preview window (see exactly what GR00T receives)

```bash
python pipeline/linux_imx274_player.py --camera-mode 1 --no-robot --preview --exposure 0.3
```

Opens two windows: the raw demosaiced feed and the sRGB uint8 frame the model sees.

### Full pipeline — camera + GR00T + robot arm

```bash
python pipeline/linux_imx274_player.py \
    --camera-mode 1 \
    --policy-host localhost \
    --policy-port 5555 \
    --lang-instruction "Move the blue dice" \
    --exposure 0.3
```

### Full pipeline with preview + CSV logging

```bash
python pipeline/linux_imx274_player.py \
    --camera-mode 1 \
    --lang-instruction "Move the blue dice" \
    --exposure 0.3 \
    --preview \
    --action-log /home/latticeapp/actions.csv \
    --sent-log /home/latticeapp/sent_commands.csv
```

CSV files land in `$HOME` which is bind-mounted, so they persist after the container exits.

### Arguments — linux_imx274_player.py

#### Camera

| Argument | Default | Description |
|---|---|---|
| `--camera-mode` | `0` | Resolution mode. `0` = 4K, `1` = 1080p, `2` = 4K alt. |
| `--hololink` | `192.168.0.2` | IP address of the HSB board. |
| `--expander-configuration` | `0` | I2C expander config (`0` or `1`). |
| `--pattern` | off | Display a built-in test pattern (0–11) instead of live camera. |

#### Display

| Argument | Default | Description |
|---|---|---|
| `--headless` | off | Run without any display window. |
| `--fullscreen` | off | Open the Holoviz window in fullscreen. |
| `--preview` | off | Open a second window showing the exact RGB uint8 frame sent to GR00T. Runs `SrgbConvertOp` on GPU — no extra CPU copy. |
| `--exposure` | `0.3` | Linear brightness multiplier applied before the sRGB curve. |

#### Robot control

| Argument | Default | Description |
|---|---|---|
| `--no-robot` | off | Skip all robot and policy code. Camera and display only. |
| `--robot-port` | `/dev/ttyACM0` | Serial port for the SO-101 follower arm. |
| `--robot-id` | `my_awesome_follower_arm` | Calibration ID — must match the filename in `~/.cache/huggingface/lerobot/calibration/robots/so_follower/`. |
| `--policy-host` | `localhost` | Hostname where the GR00T policy server is running. |
| `--policy-port` | `5555` | ZMQ port of the GR00T policy server. |
| `--action-horizon` | `8` | Steps to execute from each inference chunk before requesting a new one. |
| `--lang-instruction` | `"Move the blue dice"` | Natural language task description sent to the model. |

#### Logging

| Argument | Default | Description |
|---|---|---|
| `--action-log` | off | CSV path. Logs predicted joint goal positions before each `sync_write`. |
| `--sent-log` | off | CSV path. Logs sent commands + present positions read back after each `sync_write`. |
| `--frame-limit` | off | Exit after N frames. Useful for quick tests. |
| `--log-level` | `20` | Python logging level. `10` = DEBUG, `20` = INFO, `30` = WARNING. |

---

## Dataset recording

Recording uses two separate processes — one inside Docker (camera), one on the host (dataset writer).

### Step 1 — Start the ZMQ camera server (inside Docker)

```bash
./run.sh python pipeline/imx274_zmq_server.py --camera-mode 1 --headless
```

This runs the full IMX274 → `SrgbConvertOp` pipeline and publishes every
RGB uint8 frame as a JPEG over ZMQ on port 5556. No robot arm is needed here.

### Step 2 — Start the recorder (host, new terminal)

```bash
python recording/imx274_lerobot_record.py \
    --repo-id my_user/my_dataset \
    --task "Pick the red cube" \
    --follower-port /dev/ttyACM0 \
    --follower-id my_follower \
    --leader-port /dev/ttyACM1 \
    --leader-id my_leader \
    --num-episodes 20 \
    --episode-time 30 \
    --no-push
```

Omit `--leader-port` to record without teleoperation (follower holds position, action = state).

### Keyboard controls during recording

| Key | Action |
|---|---|
| Right arrow `→` | End current phase early and move on |
| Left arrow `←` | Discard current episode and re-record it |
| Escape | Save current episode and stop recording |

### Arguments — imx274_zmq_server.py

| Argument | Default | Description |
|---|---|---|
| `--camera-mode` | `0` | Same as main player. |
| `--hololink` | `192.168.0.2` | HSB board IP. |
| `--headless` | on | Run without display (default for server mode). |
| `--no-headless` | — | Show a preview window while streaming. |
| `--preview` | off | Open sRGB preview window (requires `--no-headless`). |
| `--zmq-port` | `5556` | ZMQ PUB port. Must match `--zmq-port` in the recorder. |
| `--camera-name` | `front` | Camera key in the ZMQ message. |
| `--jpeg-quality` | `90` | JPEG quality for ZMQ stream (1–100). |
| `--exposure` | `0.3` | Same as main player. |
| `--frame-limit` | off | Exit after N frames. |

### Arguments — imx274_lerobot_record.py

| Argument | Default | Description |
|---|---|---|
| `--repo-id` | required | Dataset repo ID, e.g. `my_user/my_dataset`. |
| `--task` | required | Single-sentence task description. |
| `--root` | HF_LEROBOT_HOME | Local directory to write the dataset. |
| `--num-episodes` | `10` | Total episodes to record. |
| `--fps` | `30` | Target recording frame rate. |
| `--episode-time` | `30` | Seconds of data per episode. |
| `--reset-time` | `10` | Seconds to reset the scene between episodes. |
| `--vcodec` | `h264_nvenc` | Video codec. `h264_nvenc` uses Jetson hardware encoder. |
| `--zmq-host` | `localhost` | Host running `imx274_zmq_server.py`. |
| `--zmq-port` | `5556` | ZMQ port (must match server). |
| `--camera-name` | `front` | Camera name in the ZMQ stream. |
| `--follower-port` | `/dev/ttyACM0` | Serial port for SO-101 follower arm. |
| `--follower-id` | `my_follower` | Follower calibration ID. |
| `--calibration-dir` | `~/.cache/huggingface/lerobot/calibration/robots/so_follower` | Directory with `<id>.json` calibration files. |
| `--leader-port` | off | Serial port for SO-101 leader arm (teleop). Omit to disable. |
| `--leader-id` | `my_leader` | Leader calibration ID. |
| `--no-push` | off | Skip pushing the finished dataset to Hugging Face Hub. |
| `--private` | off | Make the Hub repository private. |
| `--resume` | off | Resume recording into an existing dataset. |

---

## Camera modes

| `--camera-mode` | Resolution | Notes |
|---|---|---|
| `0` | 3840×2160 (4K) | Default |
| `1` | 1920×1080 (1080p) | Recommended for robot control and recording |
| `2` | 3840×2160 (4K alt) | |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'holoscan'`**
Exit the container and re-run `./run.sh`. The script sets `PYTHONPATH` automatically.

**`ModuleNotFoundError: No module named 'lerobot'` / `'cv2'`**
Run `bash install_robot_deps.sh` inside the container.

**`ModuleNotFoundError: No module named 'gr00t'`**
`run.sh` adds `~/Isaac-GR00T` to `PYTHONPATH`. Verify that directory exists on the host.

**ZMQ camera timeout / no frames received**
Ensure `imx274_zmq_server.py` is running inside Docker before starting the recorder.
Check that port 5556 is not blocked. Try `--zmq-host <jetson-ip>` if running the
recorder on a different machine.

**`Failed to initialize glfw` / X11 errors**
Run `xhost +` on the host before launching the container.

**`FeetechMotorsBus: Missing motor IDs`**
Check the arm is powered on, USB cable connected, and the port is correct (`ls /dev/ttyACM*`).
If permission denied: `chmod 666 /dev/ttyACM0`.

**`ImportError: libcudss.so.0`** (policy server)
Use the full `LD_LIBRARY_PATH` prefix shown in the setup section. Do not use conda envs.

**Image too bright or too dark**
Adjust `--exposure`. Default is `0.3`. The value multiplies the linear sensor
data before the sRGB curve, so colour relationships stay correct.

**`ERROR: v4l2loopback device not found`** (legacy v4l2 script only)
Create the device on the host first:
`sudo modprobe v4l2loopback devices=1 video_nr=10 card_label=HolovizBridge`
