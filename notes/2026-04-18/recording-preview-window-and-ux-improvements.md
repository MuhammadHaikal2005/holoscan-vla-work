# Recording Preview Window and UX Improvements
**Date:** 2026-04-18  
**Session goal:** Fix remaining recording bugs, add a live camera preview window to the dataset recorder, and improve the overall usability of the recording script.

---

## 1. Fixed `install_robot_deps.sh` crash

### What
The script crashed with `install_robot_deps.sh: line 22: on-headless:: command not found` immediately after the pip install completed.

### Why it happened
The comment `# opencv-python-headless: used by...` ended with a colon-suffix on a token (`opencv-python-headless:`). Bash with `set -e` (exit on any error) misinterpreted this as a label/colon-command rather than a comment, causing an immediate exit.

### How it was fixed
- Rewrote all comments in the pip install block to avoid `: ` at the end of a word
- Dropped the `-e` flag from `set -euo pipefail`, keeping `-uo` (catches unbound variables and pipe failures, but not every non-zero exit)
- Added `--root-user-action=ignore` to suppress the unrelated root-user pip warning

---

## 2. Persistent Docker packages

### What
Each `./run.sh` call starts a fresh `--rm` container. pip packages installed inside the container (e.g. `cv2`, `pyzmq`) were lost when the container exited, requiring `install_robot_deps.sh` to be re-run every session.

### Why it happened
The container is ephemeral. `pip3 install` targets `/usr/local/lib/...` inside the container filesystem, which is discarded when the container exits.

### How it was fixed
- Changed `install_robot_deps.sh` to install with `--target /home/latticeapp/.docker-packages`
- `/home/latticeapp` is already bind-mounted from the host (`-v $HOME:$HOME` in `run.sh`), so the packages land on persistent host storage
- Added `/home/latticeapp/.docker-packages` to `PYTHONPATH` inside `run.sh`

**Result:** `install_robot_deps.sh` is now truly a one-time operation.

---

## 3. Arm torque modes fix (both arms were locking up)

### What
Both robot arms became stiff and unresponsive immediately when recording started, making teleoperation impossible.

### Why it happened
`_create_and_connect_bus` was called identically for both the leader and follower arm. Both ended up in position-control mode with torque implicitly on, so both resisted being moved.

### How it was fixed
Added an `is_leader` parameter to `_create_and_connect_bus`:

| Arm | is_leader | Torque | Behaviour |
|---|---|---|---|
| Follower | `False` | ON | Holds and tracks commanded positions |
| Leader | `True` | OFF | Fully backdriveable — human moves it freely |

The follower also reads its current position before enabling torque, then sets that as the goal position, so it does not jump when torque kicks in.

---

## 4. Interactive prompt when dataset folder already exists

### What
If a dataset folder from a previous session was present and `--resume` was not passed, the script crashed with `FileExistsError`. Users had to remember to either delete the folder or add `--resume`, which was easy to forget.

### How it was fixed
Instead of crashing, the script now shows an interactive prompt:

```
────────────────────────────────────────────────────────────────────
  ⚠  Dataset folder already exists: .../datasets/control_blue_only
     Contains 12 recorded episode(s).
────────────────────────────────────────────────────────────────────
  [d]  Delete it and start a fresh recording session
  [r]  Resume — keep existing episodes and continue recording
  [q]  Quit
```

Choosing `r` sets `args.resume = True` programmatically, so the rest of the code path is identical to passing `--resume` on the command line.

---

## 5. Episode count prompt

### What
The target episode count was hardcoded to 100 per dataset. There was no easy way to record a smaller set (e.g. 30 episodes for a quick test) without modifying the script or passing a CLI flag.

### How it was fixed
After dataset selection, the script now asks:

```
  How many episodes do you want to record? [default: 100]:
```

- Press Enter → use the config default (100)
- Type any positive integer → use that number
- Pass `--num-episodes N` on the command line → skip the prompt entirely

---

## 6. Live camera preview window

### Why
When recording robot demonstrations, it is essential to see exactly what the camera sees — to verify framing, lighting, and that the correct object is visible — without switching to a separate terminal or display.

### First attempt — `opencv-python-headless`
Added `cv2.imshow` calls. Immediately crashed with:
```
cv2.error: The function is not implemented. Rebuild the library with GTK+ 2.x or Cocoa support.
```
**Cause:** `opencv-python-headless` intentionally strips all GUI functions.  
**Fix:** Installed `opencv-python` (full build with Qt backend) in the `lerobot2` environment.

### Second attempt — window scaling blur
Text in the preview window looked blurry. The root cause was that the canvas was `2210 × 1080` pixels (1920 camera + 290 panel) but the window was rendered at ~500px wide on screen — a 4× downscale that blurs everything regardless of font quality.

**Fix:** Resize the camera frame to `_DISPLAY_H` (default 600px) before compositing. The canvas is built at the exact pixel size it will be displayed at, so no window scaling occurs.

```python
_DISPLAY_H = 600  # change this to adjust window size
```

### Third attempt — OpenCV font quality
Even at the correct display size, `cv2.FONT_HERSHEY_SIMPLEX` looked rough. This font is a low-resolution stroke font designed for fast rendering in the 1990s, not quality display.

**Fix:** Switched panel text rendering to **Pillow (PIL)** with TrueType fonts. DejaVu Sans is available on the system at:
- Regular: `/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf`
- Bold: `/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf`

Pillow draws the side panel as a PIL Image, which is then converted to a BGR numpy array and concatenated with the cv2 camera frame for display. OpenCV is only used for window management; all text is drawn by Pillow.

### Final design

The window shows the camera feed on the left and a fixed-width info panel on the right:

```
┌──────────────────────────────────┬──────────────────┐
│                                  │ RECORDING        │
│                                  │ ────────────     │
│       Live camera feed           │ EPISODE          │
│                                  │ 3 / 100          │
│                                  │ [████░░░░░░░░░]  │
│                                  │ ────────────     │
│                                  │ FRAMES CAPTURED  │
│                                  │ 412              │
│                                  │ ────────────     │
│                                  │ TASK             │
│                                  │ Pick up the cube │
│                                  │ ────────────     │
│                                  │ KEYBOARD CTRL    │
│                                  │ Right arrow ->   │
│                                  │  Save & continue │
│                                  │ Left arrow  <-   │
│                                  │  Discard & redo  │
│                                  │ Escape           │
│                                  │  Save & stop     │
│                                  │                  │
│                                  │ Press H for help │
└──────────────────────────────────┴──────────────────┘
```

**Status colours:**
- `RECORDING` — red
- `RESET` — yellow
- Idle — grey

**Progress bar** fills left-to-right in the status colour as episodes are completed.

### Help overlay (H key)

Pressing `H` fades the camera to dark and shows a full-screen help overlay with detailed plain-English descriptions of each keyboard control. Recording continues in the background. Press `H` again to dismiss.

### Graceful fallback

Before connecting arms or creating the dataset, the script opens a test window. If `cv2.imshow` throws a `cv2.error` (e.g. no display available), the preview is automatically disabled with a warning message instead of crashing mid-recording.

Preview is **on by default**. Use `--no-preview` to disable it (e.g. for headless/remote sessions).

---

## 7. Follower arm recalibration

The follower arm was recalibrated twice during this session. The first attempt had `elbow_flex: min=815, max=3030` which appeared suspicious (POS was nearly at MAX). The second calibration produced a clean result:

```
shoulder_pan:   818 – 3514  (2696 ticks)
shoulder_lift:  907 – 3330  (2423 ticks)
elbow_flex:     738 – 2963  (2225 ticks)
wrist_flex:     886 – 3252  (2366 ticks)
gripper:       2019 – 3568  (1549 ticks)
```

Calibration command:
```bash
lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_awesome_follower_arm
```

After calibrating, copy into the repo:
```bash
cp ~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_awesome_follower_arm.json \
   ~/hsb-groot-robot/calibration/robots/so_follower/my_awesome_follower_arm.json
```

---

## 8. Notes folder reorganisation

The `notes/` folder was reorganised from flat date-prefixed filenames into date subfolders:

```
notes/
├── 2026-04-16/
│   ├── hsb-udp-payload-sender-debugging.md
│   └── zmq-recording-pipeline-and-dataset-setup.md
└── 2026-04-18/
    └── recording-preview-window-and-ux-improvements.md
```

This makes it easier to find notes by session when more dates accumulate.

---

## Summary of all files changed this session

| File | Change |
|---|---|
| `install_robot_deps.sh` | Fixed bash crash; persistent packages to `~/.docker-packages` |
| `run.sh` | Added `~/.docker-packages` to `PYTHONPATH` |
| `recording/imx274_lerobot_record.py` | Torque fix, dataset prompt, episode prompt, full preview window |
| `calibration/robots/so_follower/my_awesome_follower_arm.json` | Fresh calibration |
| `notes/` | Reorganised into date subfolders; this file added |
