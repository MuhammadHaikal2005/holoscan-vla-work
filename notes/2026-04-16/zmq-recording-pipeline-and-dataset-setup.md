# ZMQ Recording Pipeline, Dataset Setup, and Arm Calibration
**Date:** 2026-04-16  
**Session goal:** Get the IMX274 camera feeding directly into a LeRobot dataset recorder via ZMQ, fix arm torque modes, calibrate both arms, and set up the VLA generalisation experiment.

---

## Overview

The previous approach used `pyfakewebcam` to bridge the Holoscan IMX274 pipeline into LeRobot. This introduced lag because of YUYV422 conversion and kernel buffering. This session replaced it with a two-process ZMQ architecture:

- **Inside Docker** — `pipeline/imx274_zmq_server.py` runs the full Holoscan pipeline and publishes JPEG-encoded RGB frames over ZMQ (port 5556).
- **On the host** — `recording/imx274_lerobot_record.py` receives frames via LeRobot's built-in `ZMQCamera`, reads/writes joint positions from the SO-101 arms, and writes everything to a `LeRobotDataset` (Parquet + video).

---

## Architecture

```
Docker container                         Host (lerobot2 conda env)
────────────────                         ──────────────────────────
imx274_zmq_server.py                     imx274_lerobot_record.py
  LinuxReceiverOp                          ZMQCamera(localhost:5556)
    CsiToBayerOp                             ↑ JPEG over ZMQ
      ImageProcessorOp             FeetechMotorsBus — follower (/dev/ttyACM0)
        BayerDemosaicOp            FeetechMotorsBus — leader   (/dev/ttyACM1)
          SrgbConvertOp (GPU→CPU)      LeRobotDataset.add_frame()
            ZmqPublisherOp ──────→       └─ Parquet + h264_nvenc video
```

The GPU→CPU copy happens once per frame inside `ZmqPublisherOp` (`cp.asnumpy`). At 30 fps this is ~2–3 ms and is well within budget. Frames that cannot be sent without blocking are dropped with `zmq.NOBLOCK` rather than stalling the pipeline.

---

## Persistent Docker packages

Each `./run.sh` call starts a fresh `--rm` container. Previously, `pip install` inside the container was lost on exit.

**Fix:** `install_robot_deps.sh` now installs to `~/.docker-packages`:
```bash
pip3 install --target /home/latticeapp/.docker-packages \
    pyzmq msgpack pyserial deepdiff tqdm feetech-servo-sdk opencv-python-headless
```

`run.sh` adds `/home/latticeapp/.docker-packages` to `PYTHONPATH` inside the container. Since `/home/latticeapp` is bind-mounted, the packages survive container restarts. **Run `install_robot_deps.sh` once and never again** (unless updating a package).

---

## Arm torque modes

| Arm | Torque | Reason |
|---|---|---|
| **Follower** (`/dev/ttyACM0`) | ON — position tracking | Must hold and move to commanded positions |
| **Leader** (`/dev/ttyACM1`) | OFF — compliant | Human moves it freely; we only read its positions |

Before enabling torque on the follower, the script reads its current position and writes that as the goal position so it does not jump when torque kicks in.

The bug before this fix: `_create_and_connect_bus` was called identically for both arms, leaving both in stiff position-control mode. Both arms locked up as soon as recording started.

**Relevant code in `recording/imx274_lerobot_record.py`:**
```python
follower_bus = _create_and_connect_bus(port, cal, "follower", is_leader=False)
leader_bus   = _create_and_connect_bus(port, cal, "leader",   is_leader=True)
```

---

## Dataset experiment design

Two datasets for a VLA generalisation experiment, both using the same task prompt: **"Pick up the cube"**.

| Dataset | Folder | Description |
|---|---|---|
| `control_blue_only` | `datasets/control_blue_only/` | Only the blue dice in every episode. Baseline. |
| `experiment_mixed_8020` | `datasets/experiment_mixed_8020/` | Both objects present; ~80% blue dice, ~20% red cube. |

### Hypothesis
The control dataset tests whether the model can reliably pick up the blue dice. The experiment dataset tests whether mixed-colour training with a generic prompt lets the VLM generalise object selection (blue dice vs red cube) without per-colour fine-tuning.

### Why the same prompt for both objects?
The VLM component of GR00T already has broad object knowledge. The action head is what needs to learn. If the prompt always says "Pick up the cube" and both colours appear, the model is forced to use visual context to decide which object to pick up — this prevents language grounding collapse (where the action head ignores language conditioning entirely because it is never forced to disambiguate).

---

## Follower arm calibration

The follower arm was recalibrated twice in this session.

**Command:**
```bash
lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_awesome_follower_arm
```

**Final calibration (used):**
```
shoulder_pan:   818 – 3514  (2696 ticks)
shoulder_lift:  907 – 3330  (2423 ticks)
elbow_flex:     738 – 2963  (2225 ticks)
wrist_flex:     886 – 3252  (2366 ticks)
gripper:       2019 – 3568  (1549 ticks)
```

After calibration, copy the file into the repo:
```bash
cp ~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_awesome_follower_arm.json \
   ~/hsb-groot-robot/calibration/robots/so_follower/my_awesome_follower_arm.json
```

**Leader arm calibration** (done earlier, already in repo):
```
shoulder_pan:   736 – 3514  (2778 ticks)
shoulder_lift:  906 – 3371  (2465 ticks)
elbow_flex:     808 – 3071  (2263 ticks)
wrist_flex:     765 – 3173  (2408 ticks)
gripper:       2002 – 3392  (1390 ticks)
```

Calibration files live in:
```
calibration/
├── robots/so_follower/my_awesome_follower_arm.json
└── teleoperators/so_leader/my_leader.json
```

---

## Bugs fixed this session

### 1. `install_robot_deps.sh: line 22: on-headless:: command not found`
**Cause:** A comment `# opencv-python-headless: used by...` had a colon-suffix after a word at the end of the line. Combined with `set -e`, bash misinterpreted it as a label/command.  
**Fix:** Rewrote comments in the pip block to avoid `: ` at line ends. Dropped `-e` from `set -euo pipefail` (kept `-uo`).

### 2. `ModuleNotFoundError: No module named 'cupy'` when running ZMQ server
**Cause:** User ran `python pipeline/imx274_zmq_server.py` directly on the host. `cupy` only exists inside the Docker image.  
**Fix:** Always use `./run.sh python pipeline/imx274_zmq_server.py ...`

### 3. `FileExistsError` on `LeRobotDataset.create`
**Cause:** `datasets/control_blue_only/` already existed (from `.gitkeep` or a previous recording session).  
**Fix 1:** Added cleanup logic that removes the directory if it only contains a `.gitkeep` before calling `create`.  
**Fix 2 (this session):** Removed a `root.mkdir(parents=True, exist_ok=True)` call that was re-creating the directory immediately after the cleanup, before `create` was called.

### 4. Both arms locked/stiff during recording
**Cause:** `_create_and_connect_bus` was identical for leader and follower — both ended up in position-control mode with torque implicitly on.  
**Fix:** Added `is_leader` parameter. Leader: torque stays off. Follower: reads current position, sets it as goal, then enables torque.

### 5. Elbow flex calibration appeared bad (first attempt)
**First attempt result:** `elbow_flex | 815 | 3025 | 3030` — MAX and POS were almost identical.  
**Cause:** The elbow was not moved away from its maximum during the "move through full range" phase.  
**Fix:** Re-ran calibration and moved the elbow fully from extended to folded. Result: `738 – 2963` (2225 ticks).

---

## Interactive dataset prompt (new UX)

If the dataset folder already exists and `--resume` was not passed, instead of crashing the script now shows:

```
────────────────────────────────────────────────────────────────────
  ⚠  Dataset folder already exists: .../datasets/control_blue_only
     Contains 12 recorded episode(s).
────────────────────────────────────────────────────────────────────
  [d]  Delete it and start a fresh recording session
  [r]  Resume — keep existing episodes and continue recording
  [q]  Quit

  Choice [d/r/q]:
```

`--resume` still works as a flag to skip the prompt entirely.

---

## Recording workflow (end-to-end)

```bash
# Terminal 1 — camera server inside Docker
./run.sh python pipeline/imx274_zmq_server.py --camera-mode 1 --headless

# Terminal 2 — recorder on host
conda activate lerobot2
python recording/imx274_lerobot_record.py \
  --follower-port /dev/ttyACM0 \
  --follower-id my_awesome_follower_arm \
  --leader-port /dev/ttyACM1 \
  --leader-id my_leader \
  --no-push
```

The script shows an interactive menu to select which experiment dataset to record into, and prints scene setup instructions before each session.

### Keyboard controls
| Key | Action |
|---|---|
| `→` right arrow | End current phase early |
| `←` left arrow | Discard episode and re-record |
| `Esc` | Save current episode and stop |
