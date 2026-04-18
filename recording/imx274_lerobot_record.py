#!/usr/bin/env python3
# Runs on the HOST (outside Docker).
#
# Records a LeRobot-compatible dataset using:
#   - IMX274 camera frames received over ZMQ from imx274_zmq_server.py (in Docker)
#   - SO-101 follower arm connected via USB for observation.state
#   - SO-101 leader arm connected via USB for teleop actions  (optional)
#
# Dataset output: Parquet + video in LeRobot v2 format, ready for GR00T training.
#
# Typical workflow
# ----------------
# 1. In one terminal (inside Docker):
#      ./run.sh python pipeline/imx274_zmq_server.py --camera-mode 1 --headless
#
# 2. In another terminal (on host):
#      python recording/imx274_lerobot_record.py \
#        --follower-port /dev/ttyACM0 \
#        --follower-id my_follower    \
#        --leader-port /dev/ttyACM1  \
#        --leader-id my_leader
#
#   The script will interactively ask which dataset you are recording.
#   To skip the menu and go directly to a dataset, pass --dataset 1 or --dataset 2.
#
# Keyboard controls (pynput — requires non-headless terminal)
# -----------------------------------------------------------
#   Right arrow  → end episode early and save it
#   Left arrow   → discard current episode and re-record
#   Escape       → save current episode and stop recording entirely
#
# Each episode records for --episode-time seconds, then automatically saves.
# After each episode the script waits in RESET state until → is pressed.

import argparse
import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Suppress Qt "cannot find font directory" warnings that opencv-python emits
# on systems where the conda Qt fonts are missing (harmless, display still works).
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts.warning=false")

from lerobot.cameras.zmq import ZMQCamera, ZMQCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

_REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_FOLLOWER_CALIBRATION_DIR = str(
    _REPO_ROOT / "calibration" / "robots" / "so_follower"
)
DEFAULT_LEADER_CALIBRATION_DIR = str(
    _REPO_ROOT / "calibration" / "teleoperators" / "so_leader"
)

# Root directory for local dataset storage (relative to repo root)
DATASETS_ROOT = _REPO_ROOT / "datasets"

# ---------------------------------------------------------------------------
# Experiment dataset definitions
# ---------------------------------------------------------------------------
# Both datasets use the same task prompt intentionally — this is the experiment.
# Dataset 1 (control):    blue dice only, no red cube present.
# Dataset 2 (experiment): both objects present, ~80% blue dice / ~20% red cube.
# The VLM is expected to generalise object identity from the same motion data.

TASK_PROMPT = "Pick up the cube"

DATASETS = {
    "1": {
        "label":       "CONTROL — blue dice only",
        "folder":      "control_blue_only",
        "repo_id":     "latticeapp/control-blue-only",
        "description": (
            "Control dataset. Only the blue dice is present in every episode.\n"
            "Tests whether the model can pick up the blue dice reliably.\n"
            "Hypothesis: baseline performance with no colour/object variation."
        ),
        "scene_setup": [
            "  • Place the BLUE DICE anywhere on the table.",
            "  • The RED CUBE must NOT be visible in the scene.",
            "  • Vary the dice position and orientation between episodes.",
        ],
        "num_episodes": 100,
    },
    "2": {
        "label":       "EXPERIMENT — mixed 80 / 20",
        "folder":      "experiment_mixed_8020",
        "repo_id":     "latticeapp/experiment-mixed-8020",
        "description": (
            "Experiment dataset. Both objects are present in every episode.\n"
            "~80 % of episodes: pick the blue dice.\n"
            "~20 % of episodes: pick the red cube.\n"
            "Hypothesis: mixed colour data + generic prompt lets the VLM\n"
            "            generalise object selection without per-colour training."
        ),
        "scene_setup": [
            "  • Place BOTH objects on the table for every episode.",
            "  • ~80 % of episodes → pick the BLUE DICE (ignore red cube).",
            "  • ~20 % of episodes → pick the RED CUBE (ignore blue dice).",
            "  • Vary positions of both objects between episodes.",
            "  • The task prompt is identical regardless of which you pick.",
        ],
        "num_episodes": 100,
    },
}


# ---------------------------------------------------------------------------
# Motor bus helpers (mirrors linux_imx274_player.py)
# ---------------------------------------------------------------------------

def _create_and_connect_bus(port, calibration_path, label="arm", is_leader=False):
    """Create, configure and connect a FeetechMotorsBus for an SO-101 arm.

    is_leader=True  → torque stays OFF after setup; the human moves it freely.
    is_leader=False → torque is enabled after setup; the arm holds/follows positions.
    """
    import json

    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode

    norm_body = MotorNormMode.DEGREES
    motors = {
        "shoulder_pan":  Motor(1, "sts3215", norm_body),
        "shoulder_lift": Motor(2, "sts3215", norm_body),
        "elbow_flex":    Motor(3, "sts3215", norm_body),
        "wrist_flex":    Motor(4, "sts3215", norm_body),
        "wrist_roll":    Motor(5, "sts3215", norm_body),
        "gripper":       Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    }

    calibration = None
    if calibration_path and os.path.exists(calibration_path):
        with open(calibration_path) as f:
            raw = json.load(f)
        calibration = {name: MotorCalibration(**data) for name, data in raw.items()}
        logging.info("%s: loaded calibration from %s", label, calibration_path)
    else:
        logging.warning("%s: no calibration file at %s", label, calibration_path)

    bus = FeetechMotorsBus(port=port, motors=motors, calibration=calibration)
    bus.connect()
    if bus.calibration:
        bus.write_calibration(bus.calibration)

    with bus.torque_disabled():
        bus.configure_motors()
        for motor in bus.motors:
            bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
            if not is_leader:
                # Follower: stiff position tracking
                bus.write("P_Coefficient", motor, 16)
                bus.write("I_Coefficient", motor, 0)
                bus.write("D_Coefficient", motor, 32)
            else:
                # Leader: soft/backdriveable — low gains so it doesn't fight back
                bus.write("P_Coefficient", motor, 8)
                bus.write("I_Coefficient", motor, 0)
                bus.write("D_Coefficient", motor, 0)
            if motor == "gripper":
                bus.write("Max_Torque_Limit", motor, 500)
                bus.write("Protection_Current", motor, 250)
                bus.write("Overload_Torque", motor, 25)

    if is_leader:
        # Leader arm must stay compliant so the operator can move it freely.
        # Torque stays off — we only ever READ positions from it.
        for motor in bus.motors:
            bus.write("Torque_Enable", motor, 0)
        logging.info("%s: torque OFF (leader/compliant mode) on %s", label, port)
    else:
        # Follower: read current position first, set it as the goal, THEN
        # enable torque so the arm doesn't jump when recording starts.
        current = bus.sync_read("Present_Position")
        bus.sync_write("Goal_Position", current)
        for motor in bus.motors:
            bus.write("Torque_Enable", motor, 1)
        logging.info("%s: torque ON (follower mode) on %s", label, port)

    logging.info("%s: connected on %s", label, port)
    return bus


def _count_existing_episodes(root: Path) -> int:
    """Return the number of episodes already saved in a LeRobotDataset folder."""
    import json
    # Prefer the authoritative meta/info.json written by LeRobot
    info_path = root / "meta" / "info.json"
    if info_path.exists():
        try:
            with open(info_path) as f:
                return int(json.load(f).get("total_episodes", 0))
        except Exception:
            pass
    # Fallback: count episode chunk files
    for pattern in ("data/chunk-*/episode_*.parquet", "episodes/episode_*.parquet"):
        files = list(root.glob(pattern))
        if files:
            return len(set(p.stem.split("_")[1] for p in files if "_" in p.stem))
    return 0


def _read_joint_positions(bus, retries: int = 3) -> np.ndarray:
    """Synchronously read all 6 joint positions; returns float32 array shape (6,)."""
    for attempt in range(retries):
        try:
            pos = bus.sync_read("Present_Position")
            return np.array([pos[j] for j in JOINT_NAMES], dtype=np.float32)
        except Exception as e:
            if attempt < retries - 1:
                logging.debug("Joint read attempt %d failed: %s — retrying", attempt + 1, e)
                time.sleep(0.005)
            else:
                raise


def _write_joint_positions(bus, positions: np.ndarray) -> None:
    """Synchronously write 6 joint goal positions from a float32 array shape (6,)."""
    goal = {j: float(positions[i]) for i, j in enumerate(JOINT_NAMES)}
    bus.sync_write("Goal_Position", goal)


# ---------------------------------------------------------------------------
# Live preview window
# ---------------------------------------------------------------------------

_PREVIEW_WIN  = "IMX274 Recording Preview"
_DISPLAY_H    = 600     # height the preview window renders at — change this to taste
_PANEL_W      = 300    # side panel width at display resolution
_PANEL_BG     = (30,  30,  30)
_CLR_WHITE    = (240, 240, 240)
_CLR_DIM      = (130, 130, 130)
_CLR_RED      = (220,  60,  60)   # RGB for Pillow
_CLR_YELLOW   = (220, 200,  30)
_CLR_DIVIDER  = (60,  60,  60)

_help_visible = False   # toggled by pressing H in the preview window

# ---------------------------------------------------------------------------
# TrueType font loader (falls back to OpenCV if fonts not found)
# ---------------------------------------------------------------------------
_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
]
_FONT_BOLD_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
]

def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    paths = _FONT_BOLD_PATHS if bold else _FONT_PATHS
    for p in paths:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


# Pre-load at a few sizes used by the panel
_F_LABEL  = _load_font(11)          # section headers (DIM, all-caps)
_F_VALUE  = _load_font(18, bold=True)   # big numbers / status
_F_BODY   = _load_font(13)          # descriptions
_F_BOLD   = _load_font(13, bold=True)   # key names
_F_HINT   = _load_font(11)          # footer hint


def _pil_panel(h: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    """Create a blank Pillow panel image and its draw context."""
    img = Image.new("RGB", (_PANEL_W, h), _PANEL_BG)
    return img, ImageDraw.Draw(img)


def _draw_divider(draw: ImageDraw.ImageDraw, y: int):
    draw.line([(12, y), (_PANEL_W - 12, y)], fill=_CLR_DIVIDER, width=1)


def _build_panel(phase: str, episode: int, total: int,
                 frame_count: int, task: str, cam_h: int) -> np.ndarray:
    """Build the side info panel using Pillow TrueType fonts."""
    img, draw = _pil_panel(cam_h)

    phase_lower = phase.lower()
    if "record" in phase_lower:
        status, colour = "RECORDING", _CLR_RED
    elif "reset" in phase_lower:
        status, colour = "RESET",     _CLR_YELLOW
    else:
        status, colour = phase.upper(), _CLR_DIM

    PAD = 16
    y   = 20

    # ── Status ───────────────────────────────────────────────────
    draw.text((PAD, y), status, font=_F_VALUE, fill=colour);  y += 30
    _draw_divider(draw, y);  y += 14

    # ── Episode progress ─────────────────────────────────────────
    draw.text((PAD, y), "EPISODE", font=_F_LABEL, fill=_CLR_DIM);  y += 16
    draw.text((PAD, y), f"{episode}  /  {total}", font=_F_VALUE, fill=_CLR_WHITE);  y += 26
    # Progress bar
    bx, bw, bh = PAD, _PANEL_W - PAD * 2, 6
    draw.rectangle([bx, y, bx + bw, y + bh], fill=(55, 55, 55))
    filled = int(bw * episode / max(total, 1))
    if filled > 0:
        draw.rectangle([bx, y, bx + filled, y + bh], fill=colour)
    y += bh + 14
    _draw_divider(draw, y);  y += 14

    # ── Frames captured ──────────────────────────────────────────
    draw.text((PAD, y), "FRAMES CAPTURED", font=_F_LABEL, fill=_CLR_DIM);  y += 16
    frame_str = str(frame_count) if "record" in phase_lower else "-"
    draw.text((PAD, y), frame_str, font=_F_VALUE, fill=_CLR_WHITE);  y += 30
    _draw_divider(draw, y);  y += 14

    # ── Task ─────────────────────────────────────────────────────
    draw.text((PAD, y), "TASK", font=_F_LABEL, fill=_CLR_DIM);  y += 16
    words, line, lines = task.split(), "", []
    for w in words:
        test = (line + " " + w).strip()
        if draw.textlength(test, font=_F_BODY) > _PANEL_W - PAD * 2:
            lines.append(line.strip())
            line = w
        else:
            line = test
    if line.strip():
        lines.append(line.strip())
    for ln in lines:
        draw.text((PAD, y), ln, font=_F_BODY, fill=_CLR_WHITE);  y += 18
    y += 6
    _draw_divider(draw, y);  y += 14

    # ── Reset call-to-action (only during reset phase) ───────────
    if "reset" in phase_lower:
        LINE1_H = 16   # bold text height
        LINE2_H = 14   # body text height
        V_PAD   = 10   # top/bottom inner padding
        LINE_GAP = 4   # gap between the two lines
        box_h = V_PAD + LINE1_H + LINE_GAP + LINE2_H + V_PAD
        box_x1, box_x2 = PAD, _PANEL_W - PAD
        draw.rectangle([box_x1, y, box_x2, y + box_h],
                       fill=(60, 50, 10), outline=_CLR_YELLOW, width=1)
        draw.text((PAD + 8, y + V_PAD),
                  "PRESS  \u2192  TO CONTINUE",
                  font=_F_BOLD, fill=_CLR_YELLOW)
        draw.text((PAD + 8, y + V_PAD + LINE1_H + LINE_GAP),
                  "Reposition object & arm first",
                  font=_F_BODY, fill=_CLR_DIM)
        y += box_h + 12
        _draw_divider(draw, y);  y += 14

    # ── Keyboard controls (summary) ──────────────────────────────
    draw.text((PAD, y), "KEYBOARD CONTROLS", font=_F_LABEL, fill=_CLR_DIM);  y += 16
    if "reset" in phase_lower:
        controls = [
            ("Right arrow  ->", "Start next episode"),
            ("Escape",          "Save & stop"),
        ]
    else:
        controls = [
            ("Right arrow  ->", "Save & continue"),
            ("Left arrow   <-", "Discard & re-record"),
            ("Escape",          "Save & stop"),
        ]
    for key, desc in controls:
        draw.text((PAD, y),      key,  font=_F_BOLD, fill=_CLR_WHITE);  y += 16
        draw.text((PAD + 4, y),  desc, font=_F_BODY, fill=_CLR_DIM);   y += 18

    # ── Help hint (bottom) ────────────────────────────────────────
    draw.text((PAD, cam_h - 18), "Press H for help", font=_F_HINT, fill=_CLR_DIM)

    # Convert PIL RGB → numpy BGR for cv2
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _build_help_overlay(canvas: np.ndarray) -> np.ndarray:
    """Draw a full-canvas semi-transparent help overlay using Pillow text."""
    h, w = canvas.shape[:2]

    # Dark semi-transparent background
    bg = np.full_like(canvas, (18, 18, 18))
    overlay_np = cv2.addWeighted(bg, 0.88, canvas, 0.12, 0)

    # Composite text via Pillow on top of the darkened numpy frame
    pil_img = Image.fromarray(cv2.cvtColor(overlay_np, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(pil_img)

    _F_TITLE  = _load_font(22, bold=True)
    _F_KEY    = _load_font(15, bold=True)
    _F_DESC   = _load_font(13)

    cx = w // 2
    lx = cx - 380     # left edge of text block
    y  = 50

    def centre(text, font, colour):
        nonlocal y
        tw = draw.textlength(text, font=font)
        draw.text((cx - tw // 2, y), text, font=font, fill=colour)

    def hline():
        nonlocal y
        draw.line([(lx, y), (cx + 380, y)], fill=_CLR_DIVIDER, width=1)

    # Title
    centre("RECORDING CONTROLS  —  HELP", _F_TITLE, _CLR_WHITE);  y += 30
    hline();  y += 22

    sections = [
        (
            "Right arrow  ->    Save & continue",
            [
                "Ends the current recording phase early and saves the episode",
                "immediately. Use this when the task is complete before the",
                "timer runs out — no need to wait for the full duration.",
            ],
        ),
        (
            "Left arrow   <-    Discard & re-record",
            [
                "Throws away everything recorded in the current episode and",
                "restarts it from scratch. Use this if something went wrong —",
                "arm collision, object fell, or you want a cleaner demo.",
            ],
        ),
        (
            "Escape             Save & stop all recording",
            [
                "Saves the current episode, exits the recording loop, and",
                "finalises the dataset. You can resume later with --resume.",
            ],
        ),
        (
            "H                  Toggle this help screen",
            [
                "Press H at any time to show or hide this overlay.",
                "Recording continues in the background while help is shown.",
            ],
        ),
    ]

    for title, desc_lines in sections:
        draw.text((lx, y), title, font=_F_KEY, fill=_CLR_WHITE);   y += 24
        for line in desc_lines:
            draw.text((lx + 8, y), line, font=_F_DESC, fill=_CLR_DIM);  y += 18
        y += 10
        hline();  y += 18

    hint = "Press H to close"
    hw = draw.textlength(hint, font=_F_DESC)
    draw.text((cx - hw // 2, y + 6), hint, font=_F_DESC, fill=_CLR_DIM)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_preview(frame_bgr: np.ndarray, phase: str, episode: int,
                 total: int, frame_count: int, task: str) -> None:
    """Composite the camera feed with the side panel and display it.

    The camera frame is resized to _DISPLAY_H so that the composited canvas
    is shown at its native pixel size — no window scaling, no blur.
    Pressing H toggles the help overlay.
    """
    global _help_visible

    # Resize camera frame to display height (keep aspect ratio)
    h, w = frame_bgr.shape[:2]
    display_w = int(w * _DISPLAY_H / h)
    cam_display = cv2.resize(frame_bgr, (display_w, _DISPLAY_H), interpolation=cv2.INTER_AREA)

    # Build panel at the same height
    panel  = _build_panel(phase, episode, total, frame_count, task, _DISPLAY_H)
    canvas = np.concatenate([cam_display, panel], axis=1)

    if _help_visible:
        canvas = _build_help_overlay(canvas)

    cv2.imshow(_PREVIEW_WIN, canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('h') or key == ord('H'):
        _help_visible = not _help_visible


def close_preview() -> None:
    """Destroy the preview window if it exists."""
    try:
        cv2.destroyWindow(_PREVIEW_WIN)
        cv2.waitKey(1)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Dataset feature schema
# ---------------------------------------------------------------------------

def build_features(img_h: int, img_w: int) -> dict:
    return {
        "observation.images.front": {
            "dtype": "video",
            "shape": (img_h, img_w, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": JOINT_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": JOINT_NAMES,
        },
    }


# ---------------------------------------------------------------------------
# Keyboard listener
# ---------------------------------------------------------------------------

def init_keyboard_listener():
    """Non-blocking pynput listener; returns (listener, events_dict)."""
    events = {
        "exit_early": False,       # right arrow  — save + end phase early
        "rerecord_episode": False,  # left arrow   — discard + redo
        "stop_recording": False,    # escape       — save + quit
    }

    try:
        from pynput import keyboard

        def on_press(key):
            if key == keyboard.Key.right:
                logging.info("→ pressed: ending phase early")
                events["exit_early"] = True
            elif key == keyboard.Key.left:
                logging.info("← pressed: will rerecord episode")
                events["rerecord_episode"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.esc:
                logging.info("ESC pressed: stopping recording")
                events["stop_recording"] = True
                events["exit_early"] = True

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
    except Exception as e:
        logging.warning("Could not start keyboard listener: %s", e)
        listener = None

    return listener, events


# ---------------------------------------------------------------------------
# Core recording loop
# ---------------------------------------------------------------------------

def record_episode(
    dataset: LeRobotDataset,
    camera: ZMQCamera,
    follower_bus,
    leader_bus,           # None if no teleop
    task: str,
    fps: int,
    episode_time_s: float,
    events: dict,
    episode_idx: int = 0,
    total_episodes: int = 0,
    show_preview: bool = False,
) -> int:
    """Record a single episode into *dataset* and return the frame count.

    Each tick:
      1. Read an RGB frame from ZMQCamera.
      2. Read follower joint positions (observation.state).
      3. If leader arm: read leader positions, send them to follower (action).
         Otherwise: record a zero-velocity action (follower holds position).
      4. Call dataset.add_frame().

    Returns the number of frames captured.
    """
    dt = 1.0 / fps
    start_t = time.perf_counter()
    frame_count = 0

    while True:
        t0 = time.perf_counter()

        # Honour keyboard exit signals
        if events["exit_early"] or events["stop_recording"]:
            break

        # Honour time limit
        if (time.perf_counter() - start_t) >= episode_time_s:
            break

        # --- Camera ---
        # ZMQCamera decodes JPEG with cv2 → BGR; we need RGB for the dataset.
        frame_bgr = camera.async_read(timeout_ms=500)
        frame_rgb = frame_bgr[:, :, ::-1].copy()

        # --- Robot state (follower arm) ---
        obs_state = _read_joint_positions(follower_bus)

        # --- Action (leader arm or identity) ---
        if leader_bus is not None:
            action = _read_joint_positions(leader_bus)
            _write_joint_positions(follower_bus, action)
        else:
            # No teleop: follower holds position — record current state as action
            action = obs_state.copy()

        # --- Write to dataset ---
        dataset.add_frame({
            "observation.images.front": frame_rgb,
            "observation.state": obs_state,
            "action": action,
            "task": task,
        })

        frame_count += 1

        # --- Preview window ---
        if show_preview:
            draw_preview(frame_bgr, "Recording", episode_idx, total_episodes,
                         frame_count, task)

        elapsed = time.perf_counter() - t0
        precise_sleep(max(dt - elapsed, 0.0))

    return frame_count


def reset_phase(
    camera: ZMQCamera,
    follower_bus,
    leader_bus,
    fps: int,
    events: dict,
    episode_idx: int = 0,
    total_episodes: int = 0,
    task: str = "",
    show_preview: bool = False,
) -> None:
    """Wait during environment reset — holds indefinitely until → is pressed.

    The operator uses this time to reposition the object and move the robot
    arm back to its starting pose. Nothing is recorded. The follower arm
    continues to mirror the leader arm (if connected) so the operator can
    guide it back by hand.

    Press → to start the next episode.
    Press ESC to save the current state and stop recording entirely.
    """
    dt = 1.0 / fps

    print()
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │  RESET — reposition the object and the robot arm.   │")
    print("  │  Press  →  when ready to record the next episode.   │")
    print("  │  Press ESC to stop recording.                       │")
    print("  └─────────────────────────────────────────────────────┘")

    while not events["exit_early"] and not events["stop_recording"]:
        if leader_bus is not None:
            action = _read_joint_positions(leader_bus)
            _write_joint_positions(follower_bus, action)

        if show_preview:
            frame_bgr = camera.async_read(timeout_ms=500)
            draw_preview(frame_bgr, "Reset — Press -> to continue",
                         episode_idx, total_episodes, 0, task)
        else:
            precise_sleep(dt)


# ---------------------------------------------------------------------------
# Interactive dataset selector
# ---------------------------------------------------------------------------

def select_dataset(preselect: str | None) -> dict:
    """Print the experiment menu and return the chosen dataset config dict."""
    W = 76

    def rule():  return "╠" + "═" * W + "╣"
    def top():   return "╔" + "═" * W + "╗"
    def bot():   return "╚" + "═" * W + "╝"
    def row(s=""):
        padding = max(0, W - len(s))
        return "║" + s + " " * padding + "║"

    print()
    print(top())
    print(row("  IMX274 LeRobot Dataset Recorder"))
    print(row(f"  Task prompt (both datasets): \"{TASK_PROMPT}\""))
    print(rule())

    for key, ds in DATASETS.items():
        print(row())
        print(row(f"  [{key}]  {ds['label']}"))
        print(row())
        for line in ds["description"].splitlines():
            print(row("      " + line))
        print(row())
        print(row(f"      Saved to:  datasets/{ds['folder']}/"))
        print(row(f"      Target:    {ds['num_episodes']} episodes"))
        if key != list(DATASETS.keys())[-1]:
            print(rule())

    print(bot())
    print()

    if preselect is not None:
        choice = preselect.strip()
        if choice not in DATASETS:
            raise SystemExit(f"Invalid --dataset value '{choice}'. Choose from: {list(DATASETS.keys())}")
        print(f"Using pre-selected dataset [{choice}]: {DATASETS[choice]['label']}")
    else:
        while True:
            choice = input(f"  Select dataset [{'/'.join(DATASETS.keys())}]: ").strip()
            if choice in DATASETS:
                break
            print(f"  Please enter one of: {list(DATASETS.keys())}")

    ds = DATASETS[choice]
    print()
    print("─" * (W + 2))
    print(f"  Selected: {ds['label']}")
    print()
    print("  Scene setup instructions:")
    for line in ds["scene_setup"]:
        print(line)
    print()
    print(f"  Task prompt every episode: \"{TASK_PROMPT}\"")
    print("─" * (W + 2))
    input("\n  Press Enter when the scene is ready and the arms are connected …\n")
    return ds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Record a LeRobot dataset from the IMX274 camera via ZMQ."
    )

    # Dataset selection
    parser.add_argument("--dataset", default=None, metavar="N",
                        help="Skip the interactive menu and go straight to dataset N (1 or 2)")
    parser.add_argument("--repo-id", default=None,
                        help="Override the repo ID from the dataset config")
    parser.add_argument("--root", default=None,
                        help="Override the local root directory for the dataset")
    parser.add_argument("--num-episodes", type=int, default=None,
                        help="Override the number of episodes from the dataset config")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--episode-time", type=float, default=30.0, dest="episode_time_s",
                        help="Seconds of data per episode (default: 30)")
    # --reset-time removed: reset phase now waits indefinitely for → key press.
    parser.add_argument("--vcodec", default="h264_nvenc",
                        help="Video codec for encoding (default: h264_nvenc for Jetson). "
                             "Falls back to h264 automatically if nvenc is unavailable.")
    parser.add_argument("--no-push", action="store_true",
                        help="Do not push the finished dataset to Hugging Face Hub")
    parser.add_argument("--private", action="store_true",
                        help="Make the Hugging Face Hub repository private")

    # Camera (ZMQ)
    parser.add_argument("--zmq-host", default="localhost",
                        help="Host running imx274_zmq_server.py (default: localhost)")
    parser.add_argument("--zmq-port", type=int, default=5556,
                        help="ZMQ port (must match imx274_zmq_server.py, default: 5556)")
    parser.add_argument("--camera-name", default="front",
                        help="Camera name in the ZMQ stream (default: front)")

    # Follower arm
    parser.add_argument("--follower-port", default="/dev/ttyACM0",
                        help="Serial port for the SO-101 follower arm")
    parser.add_argument("--follower-id", default="my_follower",
                        help="Robot ID used to locate the calibration JSON")
    parser.add_argument("--calibration-dir", default=DEFAULT_FOLLOWER_CALIBRATION_DIR,
                        help="Directory containing follower <id>.json calibration files")
    parser.add_argument("--leader-calibration-dir", default=DEFAULT_LEADER_CALIBRATION_DIR,
                        help="Directory containing leader <id>.json calibration files")

    # Leader arm (teleop) — optional
    parser.add_argument("--leader-port", default=None,
                        help="Serial port for the SO-101 leader arm (omit to disable teleop)")
    parser.add_argument("--leader-id", default="my_leader",
                        help="Leader arm calibration ID (only used when --leader-port is set)")

    # Misc
    parser.add_argument("--log-level", type=int, default=20)
    parser.add_argument("--resume", action="store_true",
                        help="Resume recording on an existing dataset")
    parser.add_argument("--no-preview", action="store_true",
                        help="Disable the live camera preview window")

    args = parser.parse_args()

    init_logging()
    logging.getLogger().setLevel(args.log_level)

    # ------------------------------------------------------------------ Dataset selection
    ds_config = select_dataset(args.dataset)

    repo_id = args.repo_id or ds_config["repo_id"]
    root    = Path(args.root) if args.root else DATASETS_ROOT / ds_config["folder"]
    task    = TASK_PROMPT

    # ------------------------------------------------------------------ Folder check
    # Must run BEFORE the episode count prompt so that on resume we know how
    # many episodes already exist and can ask a meaningful question.
    if not args.resume and root.exists():
        import shutil
        contents = list(root.iterdir())
        is_placeholder = contents == [] or (len(contents) == 1 and contents[0].name == ".gitkeep")

        if is_placeholder:
            shutil.rmtree(root)
        else:
            existing_episodes = _count_existing_episodes(root)

            print()
            print("─" * 68)
            print(f"  ⚠  Dataset folder already exists: {root}")
            if existing_episodes:
                print(f"     Contains {existing_episodes} recorded episode(s).")
            print("─" * 68)
            print("  [d]  Delete it and start a fresh recording session")
            print("  [r]  Resume — keep existing episodes and continue recording")
            print("  [q]  Quit")
            print()

            while True:
                choice = input("  Choice [d/r/q]: ").strip().lower()
                if choice in ("d", "r", "q"):
                    break
                print("  Please type d, r, or q.")

            print()
            if choice == "q":
                raise SystemExit(0)
            elif choice == "d":
                shutil.rmtree(root)
                print(f"  Deleted {root}. Starting fresh.\n")
            else:
                args.resume = True
                print("  Resuming existing dataset.\n")

    # ------------------------------------------------------------------ Episode count
    if args.num_episodes:
        num_episodes = args.num_episodes
    elif args.resume:
        # On resume: show how many already exist and ask for the TOTAL target.
        existing_episodes = _count_existing_episodes(root)
        default_total = max(ds_config["num_episodes"], existing_episodes + 1)
        print()
        print(f"  {existing_episodes} episode(s) already recorded.")
        while True:
            raw = input(
                f"  Record up to how many episodes in total? [default: {default_total}]: "
            ).strip()
            if raw == "":
                num_episodes = default_total
                break
            if raw.isdigit() and int(raw) > existing_episodes:
                num_episodes = int(raw)
                break
            if raw.isdigit() and int(raw) <= existing_episodes:
                print(f"  Must be greater than the {existing_episodes} already recorded.")
            else:
                print("  Please enter a positive whole number.")
        remaining = num_episodes - existing_episodes
        print(f"  Will record {remaining} more episode(s) to reach {num_episodes} total.\n")
    else:
        default_eps = ds_config["num_episodes"]
        print()
        while True:
            raw = input(f"  How many episodes do you want to record? [default: {default_eps}]: ").strip()
            if raw == "":
                num_episodes = default_eps
                break
            if raw.isdigit() and int(raw) > 0:
                num_episodes = int(raw)
                break
            print("  Please enter a positive whole number.")
        print()

    logging.info("Dataset : %s", ds_config["label"])
    logging.info("Repo ID : %s", repo_id)
    logging.info("Root    : %s", root)
    logging.info("Episodes: %d", num_episodes)
    logging.info("Task    : %s", task)

    # ------------------------------------------------------------------ Camera
    logging.info("Connecting to ZMQ camera at %s:%d …", args.zmq_host, args.zmq_port)
    cam_config = ZMQCameraConfig(
        server_address=args.zmq_host,
        port=args.zmq_port,
        camera_name=args.camera_name,
        timeout_ms=5000,
        warmup_s=2,
    )
    camera = ZMQCamera(cam_config)
    camera.connect(warmup=True)

    img_h = camera.height
    img_w = camera.width
    logging.info("Camera resolution detected: %dx%d", img_w, img_h)

    # ------------------------------------------------------------------ Preview check
    show_preview = not args.no_preview
    if show_preview:
        try:
            # Open the window now at full size so it's ready before recording starts.
            # Use WINDOW_NORMAL so the user can resize it freely.
            # WINDOW_GUI_NORMAL removes the Qt toolbar icons.
            # WINDOW_AUTOSIZE prevents cv2 from rescaling the canvas —
            # we manage the display size ourselves via _DISPLAY_H.
            cv2.namedWindow(_PREVIEW_WIN, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
            cam_w = int(16 / 9 * _DISPLAY_H)   # 16:9 placeholder
            cam_area = np.zeros((_DISPLAY_H, cam_w, 3), dtype=np.uint8)
            # Placeholder text via Pillow so it's sharp too
            pil_tmp = Image.fromarray(cam_area)
            _f = _load_font(18)
            _d = ImageDraw.Draw(pil_tmp)
            msg = "Waiting for first frame..."
            tw = _d.textlength(msg, font=_f)
            _d.text((cam_w // 2 - tw // 2, _DISPLAY_H // 2 - 10),
                    msg, font=_f, fill=(140, 140, 140))
            cam_area = cv2.cvtColor(np.array(pil_tmp), cv2.COLOR_RGB2BGR)
            panel = _build_panel("Idle", 0, 0, 0, "...", _DISPLAY_H)
            canvas = np.concatenate([cam_area, panel], axis=1)
            cv2.imshow(_PREVIEW_WIN, canvas)
            cv2.waitKey(1)
        except cv2.error as e:
            logging.warning(
                "Preview window unavailable (%s). "
                "Install 'opencv-python' (not headless) or run with --no-preview. "
                "Continuing without preview.", e
            )
            show_preview = False

    # ------------------------------------------------------------------ Arms
    follower_cal = os.path.join(args.calibration_dir, f"{args.follower_id}.json")
    follower_bus = _create_and_connect_bus(args.follower_port, follower_cal, "follower")

    leader_bus = None
    if args.leader_port:
        leader_cal = os.path.join(args.leader_calibration_dir, f"{args.leader_id}.json")
        leader_bus = _create_and_connect_bus(args.leader_port, leader_cal, "leader", is_leader=True)

    # ------------------------------------------------------------------ Dataset
    features = build_features(img_h, img_w)

    if args.resume:
        # For resume, ensure the directory exists (LeRobotDataset needs it).
        root.mkdir(parents=True, exist_ok=True)
        dataset = LeRobotDataset(
            repo_id,
            root=str(root),
            streaming_encoding=True,
            encoder_threads=2,
        )
        dataset.start_image_writer(num_processes=0, num_threads=4)
    else:
        dataset = LeRobotDataset.create(
            repo_id,
            args.fps,
            root=str(root),
            robot_type="so101_imx274",
            features=features,
            use_videos=True,
            image_writer_processes=0,
            image_writer_threads=4,
            vcodec=args.vcodec,
            streaming_encoding=True,
            encoder_threads=2,
        )

    logging.info("Dataset root: %s", dataset.root)

    # ------------------------------------------------------------------ Keyboard
    listener, events = init_keyboard_listener()

    print("\n" + "=" * 60)
    print("  Recording controls")
    print("  Right arrow  → save episode and end phase early")
    print("  Left arrow   → discard current episode and re-record")
    print("  Escape       → save current episode and stop")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------ Episode loop
    try:
        with VideoEncodingManager(dataset):
            # On resume, start the counter at however many episodes are already
            # saved so numbering and the loop condition are both correct.
            recorded = dataset.num_episodes

            while recorded < num_episodes and not events["stop_recording"]:
                log_say(f"Recording episode {recorded + 1} of {num_episodes}", play_sounds=False)
                print(f"\n  Episode {recorded + 1} / {num_episodes}  |  task: \"{task}\"")
                logging.info("Recording for %.0fs …", args.episode_time_s)

                # Reset per-episode keyboard flags
                events["exit_early"] = False
                events["rerecord_episode"] = False

                n_frames = record_episode(
                    dataset=dataset,
                    camera=camera,
                    follower_bus=follower_bus,
                    leader_bus=leader_bus,
                    task=task,
                    fps=args.fps,
                    episode_time_s=args.episode_time_s,
                    events=events,
                    episode_idx=recorded + 1,
                    total_episodes=num_episodes,
                    show_preview=show_preview,
                )
                logging.info("Episode ended: %d frames captured.", n_frames)

                # Discard and retry?
                if events["rerecord_episode"]:
                    log_say("Discarding episode — re-recording", play_sounds=False)
                    dataset.clear_episode_buffer()
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    continue

                # Save episode
                dataset.save_episode()
                recorded += 1
                log_say(f"Episode {recorded} saved.", play_sounds=False)
                print(f"  Saved. Progress: {recorded} / {num_episodes}")

                # Reset phase (skip after last episode)
                if not events["stop_recording"] and recorded < num_episodes:
                    log_say("Reset the environment", play_sounds=False)
                    logging.info("Reset phase: waiting for -> key …")
                    events["exit_early"] = False
                    reset_phase(
                        camera=camera,
                        follower_bus=follower_bus,
                        leader_bus=leader_bus,
                        fps=args.fps,
                        events=events,
                        episode_idx=recorded,
                        total_episodes=num_episodes,
                        task=task,
                        show_preview=show_preview,
                    )

    finally:
        log_say("Stopping recording", play_sounds=False)

        if show_preview:
            close_preview()

        camera.disconnect()

        try:
            if follower_bus.is_connected:
                follower_bus.disconnect(disable_torque=True)
                logging.info("Follower arm disconnected.")
        except Exception as e:
            logging.warning("Could not cleanly disconnect follower arm: %s", e)

        try:
            if leader_bus is not None and leader_bus.is_connected:
                leader_bus.disconnect(disable_torque=False)
                logging.info("Leader arm disconnected.")
        except Exception as e:
            logging.warning("Could not cleanly disconnect leader arm: %s", e)

        if listener:
            listener.stop()

        dataset.finalize()
        logging.info("Dataset finalised at %s", dataset.root)

        if not args.no_push:
            logging.info("Pushing dataset to Hugging Face Hub …")
            dataset.push_to_hub(private=args.private)
        else:
            logging.info("Skipping Hub push (--no-push).")


if __name__ == "__main__":
    main()
