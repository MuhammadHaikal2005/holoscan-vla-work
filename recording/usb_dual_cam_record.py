#!/usr/bin/env python3
# Runs on the HOST (outside Docker). No ZMQ / Docker / IMX274 required.
#
# Records a LeRobot-compatible dataset using:
#   - Two USB cameras read directly via OpenCV (/dev/video0 and /dev/video2)
#   - SO-101 follower arm connected via USB for observation.state
#   - SO-101 leader arm connected via USB for teleop actions (optional)
#
# Dataset output: Parquet + video in LeRobot v2 format, ready for GR00T training.
#
# Typical workflow
# ----------------
#   python recording/usb_dual_cam_record.py \
#     --cam0 0 \
#     --cam1 2 \
#     --follower-port /dev/ttyACM0 \
#     --follower-id my_awesome_follower_arm \
#     --leader-port /dev/ttyACM1 \
#     --leader-id my_leader \
#     --no-push
#
# Keyboard controls
# -----------------
#   Right arrow  → end episode early and save it
#   Left arrow   → discard current episode and re-record
#   Escape       → save current episode and stop recording entirely

import argparse
import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts.warning=false")

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

DATASETS_ROOT = _REPO_ROOT / "datasets"

# ---------------------------------------------------------------------------
# Experiment dataset definitions
# ---------------------------------------------------------------------------

TASK_PROMPT = "Pick up the cube"

DATASETS = {
    "1": {
        "label":       "DUAL-CAM CONTROL — blue dice only",
        "folder":      "dual_cam_blue_only",
        "repo_id":     "latticeapp/dual-cam-blue-only",
        "description": (
            "Dual-camera control dataset. Only the blue dice is present.\n"
            "Two USB cameras provide depth cues via different angles.\n"
            "Hypothesis: stereo-like views fix depth perception failures."
        ),
        "scene_setup": [
            "  • Place the BLUE DICE anywhere on the table.",
            "  • The RED CUBE must NOT be visible in the scene.",
            "  • Vary the dice position and orientation between episodes.",
            "  • Verify both camera views are unobstructed before starting.",
        ],
        "num_episodes": 100,
    },
    "2": {
        "label":       "DUAL-CAM EXPERIMENT — mixed 80 / 20",
        "folder":      "dual_cam_mixed_8020",
        "repo_id":     "latticeapp/dual-cam-mixed-8020",
        "description": (
            "Dual-camera experiment dataset. Both objects present.\n"
            "~80 % of episodes: pick the blue dice.\n"
            "~20 % of episodes: pick the red cube.\n"
            "Hypothesis: multi-view + mixed colour data generalises object selection."
        ),
        "scene_setup": [
            "  • Place BOTH objects on the table for every episode.",
            "  • ~80 % of episodes → pick the BLUE DICE.",
            "  • ~20 % of episodes → pick the RED CUBE.",
            "  • Vary positions of both objects between episodes.",
        ],
        "num_episodes": 100,
    },
}


# ---------------------------------------------------------------------------
# USB camera helpers
# ---------------------------------------------------------------------------

class USBCamera:
    """Thin OpenCV wrapper that mirrors the ZMQCamera async_read interface."""

    def __init__(self, device_index: int, width: int = 1920, height: int = 1080):
        self.device_index = device_index
        self._cap = None
        self._width  = width
        self._height = height

    def connect(self):
        self._cap = cv2.VideoCapture(self.device_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open /dev/video{self.device_index}")
        # Request the target resolution — camera may override to nearest supported
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        # Try MJPEG for lower USB bandwidth and better frame rate
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        # Warm up — discard the first few frames (auto-exposure settling)
        for _ in range(10):
            self._cap.read()
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logging.info("Camera %d opened: %dx%d", self.device_index, actual_w, actual_h)
        self._width  = actual_w
        self._height = actual_h

    @property
    def width(self):  return self._width
    @property
    def height(self): return self._height

    def read_bgr(self) -> np.ndarray:
        """Read one frame (BGR uint8). Raises RuntimeError on failure."""
        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise RuntimeError(f"Frame read failed on device {self.device_index}")
        return frame

    def read_rgb(self) -> np.ndarray:
        return self.read_bgr()[:, :, ::-1].copy()

    def disconnect(self):
        if self._cap and self._cap.isOpened():
            self._cap.release()
            logging.info("Camera %d released.", self.device_index)


# ---------------------------------------------------------------------------
# Motor bus helpers (identical to imx274_lerobot_record.py)
# ---------------------------------------------------------------------------

def _create_and_connect_bus(port, calibration_path, label="arm", is_leader=False):
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
        import json
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
                bus.write("P_Coefficient", motor, 16)
                bus.write("I_Coefficient", motor, 0)
                bus.write("D_Coefficient", motor, 32)
            else:
                bus.write("P_Coefficient", motor, 8)
                bus.write("I_Coefficient", motor, 0)
                bus.write("D_Coefficient", motor, 0)
            if motor == "gripper":
                bus.write("Max_Torque_Limit", motor, 500)
                bus.write("Protection_Current", motor, 250)
                bus.write("Overload_Torque", motor, 25)

    if is_leader:
        for motor in bus.motors:
            bus.write("Torque_Enable", motor, 0)
        logging.info("%s: torque OFF (leader/compliant mode) on %s", label, port)
    else:
        current = bus.sync_read("Present_Position")
        bus.sync_write("Goal_Position", current)
        for motor in bus.motors:
            bus.write("Torque_Enable", motor, 1)
        logging.info("%s: torque ON (follower mode) on %s", label, port)

    logging.info("%s: connected on %s", label, port)
    return bus


def _count_existing_episodes(root: Path) -> int:
    import json
    info_path = root / "meta" / "info.json"
    if info_path.exists():
        try:
            with open(info_path) as f:
                return int(json.load(f).get("total_episodes", 0))
        except Exception:
            pass
    for pattern in ("data/chunk-*/episode_*.parquet", "episodes/episode_*.parquet"):
        files = list(root.glob(pattern))
        if files:
            return len(set(p.stem.split("_")[1] for p in files if "_" in p.stem))
    return 0


def _read_joint_positions(bus, retries: int = 3) -> np.ndarray:
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
    goal = {j: float(positions[i]) for i, j in enumerate(JOINT_NAMES)}
    bus.sync_write("Goal_Position", goal)


# ---------------------------------------------------------------------------
# Live preview window
# ---------------------------------------------------------------------------

_PREVIEW_WIN = "Dual-USB Recording Preview"
_DISPLAY_H   = 480    # height each camera feed renders at
_PANEL_W     = 300
_PANEL_BG    = (30, 30, 30)
_CLR_WHITE   = (240, 240, 240)
_CLR_DIM     = (130, 130, 130)
_CLR_RED     = (220,  60,  60)
_CLR_YELLOW  = (220, 200,  30)
_CLR_DIVIDER = (60,  60,  60)

_help_visible = False

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

def _load_font(size, bold=False):
    paths = _FONT_BOLD_PATHS if bold else _FONT_PATHS
    for p in paths:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

_F_LABEL = _load_font(11)
_F_VALUE = _load_font(18, bold=True)
_F_BODY  = _load_font(13)
_F_BOLD  = _load_font(13, bold=True)
_F_HINT  = _load_font(11)


def _build_panel(phase, episode, total, frame_count, task, cam_h):
    img  = Image.new("RGB", (_PANEL_W, cam_h), _PANEL_BG)
    draw = ImageDraw.Draw(img)

    phase_lower = phase.lower()
    if "record" in phase_lower:
        status, colour = "RECORDING", _CLR_RED
    elif "reset" in phase_lower:
        status, colour = "RESET",     _CLR_YELLOW
    else:
        status, colour = phase.upper(), _CLR_DIM

    PAD = 16
    y   = 20

    draw.text((PAD, y), status, font=_F_VALUE, fill=colour); y += 30
    draw.line([(12, y), (_PANEL_W - 12, y)], fill=_CLR_DIVIDER, width=1); y += 14

    draw.text((PAD, y), "EPISODE", font=_F_LABEL, fill=_CLR_DIM); y += 16
    draw.text((PAD, y), f"{episode}  /  {total}", font=_F_VALUE, fill=_CLR_WHITE); y += 26
    bx, bw, bh = PAD, _PANEL_W - PAD * 2, 6
    draw.rectangle([bx, y, bx + bw, y + bh], fill=(55, 55, 55))
    filled = int(bw * episode / max(total, 1))
    if filled > 0:
        draw.rectangle([bx, y, bx + filled, y + bh], fill=colour)
    y += bh + 14
    draw.line([(12, y), (_PANEL_W - 12, y)], fill=_CLR_DIVIDER, width=1); y += 14

    draw.text((PAD, y), "FRAMES CAPTURED", font=_F_LABEL, fill=_CLR_DIM); y += 16
    draw.text((PAD, y), str(frame_count) if "record" in phase_lower else "-",
              font=_F_VALUE, fill=_CLR_WHITE); y += 30
    draw.line([(12, y), (_PANEL_W - 12, y)], fill=_CLR_DIVIDER, width=1); y += 14

    draw.text((PAD, y), "TASK", font=_F_LABEL, fill=_CLR_DIM); y += 16
    words, line, lines = task.split(), "", []
    for w in words:
        test = (line + " " + w).strip()
        if draw.textlength(test, font=_F_BODY) > _PANEL_W - PAD * 2:
            lines.append(line.strip()); line = w
        else:
            line = test
    if line.strip():
        lines.append(line.strip())
    for ln in lines:
        draw.text((PAD, y), ln, font=_F_BODY, fill=_CLR_WHITE); y += 18
    y += 6
    draw.line([(12, y), (_PANEL_W - 12, y)], fill=_CLR_DIVIDER, width=1); y += 14

    if "reset" in phase_lower:
        V_PAD, LINE1_H, LINE2_H, LINE_GAP = 10, 16, 14, 4
        box_h = V_PAD + LINE1_H + LINE_GAP + LINE2_H + V_PAD
        draw.rectangle([PAD, y, _PANEL_W - PAD, y + box_h],
                       fill=(60, 50, 10), outline=_CLR_YELLOW, width=1)
        draw.text((PAD + 8, y + V_PAD), "PRESS  \u2192  TO CONTINUE",
                  font=_F_BOLD, fill=_CLR_YELLOW)
        draw.text((PAD + 8, y + V_PAD + LINE1_H + LINE_GAP),
                  "Reposition object & arm first", font=_F_BODY, fill=_CLR_DIM)
        y += box_h + 12
        draw.line([(12, y), (_PANEL_W - 12, y)], fill=_CLR_DIVIDER, width=1); y += 14

    draw.text((PAD, y), "KEYBOARD CONTROLS", font=_F_LABEL, fill=_CLR_DIM); y += 16
    controls = ([("Right arrow ->", "Start next episode"), ("Escape", "Save & stop")]
                if "reset" in phase_lower else
                [("Right arrow ->", "Save & continue"),
                 ("Left arrow  <-", "Discard & re-record"),
                 ("Escape",         "Save & stop")])
    for key, desc in controls:
        draw.text((PAD, y), key,  font=_F_BOLD, fill=_CLR_WHITE); y += 16
        draw.text((PAD + 4, y), desc, font=_F_BODY, fill=_CLR_DIM); y += 18

    draw.text((PAD, cam_h - 18), "Press H for help", font=_F_HINT, fill=_CLR_DIM)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def draw_preview(frame0_bgr, frame1_bgr, phase, episode, total, frame_count, task):
    global _help_visible

    def _resize(f):
        h, w = f.shape[:2]
        new_w = int(w * _DISPLAY_H / h)
        return cv2.resize(f, (new_w, _DISPLAY_H), interpolation=cv2.INTER_AREA)

    cam0 = _resize(frame0_bgr)
    cam1 = _resize(frame1_bgr)
    panel = _build_panel(phase, episode, total, frame_count, task, _DISPLAY_H)

    # Stack: [cam0 | cam1 | panel]
    canvas = np.concatenate([cam0, cam1, panel], axis=1)
    cv2.imshow(_PREVIEW_WIN, canvas)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('h'), ord('H')):
        _help_visible = not _help_visible


def close_preview():
    try:
        cv2.destroyWindow(_PREVIEW_WIN)
        cv2.waitKey(1)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Dataset feature schema
# ---------------------------------------------------------------------------

def build_features(h0, w0, h1, w1):
    return {
        "observation.images.cam0": {
            "dtype": "video",
            "shape": (h0, w0, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.cam1": {
            "dtype": "video",
            "shape": (h1, w1, 3),
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
    events = {
        "exit_early":       False,
        "rerecord_episode": False,
        "stop_recording":   False,
    }
    try:
        from pynput import keyboard

        def on_press(key):
            if key == keyboard.Key.right:
                events["exit_early"] = True
            elif key == keyboard.Key.left:
                events["rerecord_episode"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.esc:
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
    dataset, cam0, cam1, follower_bus, leader_bus,
    task, fps, episode_time_s, events,
    episode_idx=0, total_episodes=0, show_preview=False,
):
    dt = 1.0 / fps
    start_t = time.perf_counter()
    frame_count = 0

    while True:
        t0 = time.perf_counter()

        if events["exit_early"] or events["stop_recording"]:
            break
        if (time.perf_counter() - start_t) >= episode_time_s:
            break

        frame0_bgr = cam0.read_bgr()
        frame1_bgr = cam1.read_bgr()
        frame0_rgb = frame0_bgr[:, :, ::-1].copy()
        frame1_rgb = frame1_bgr[:, :, ::-1].copy()

        obs_state = _read_joint_positions(follower_bus)

        if leader_bus is not None:
            action = _read_joint_positions(leader_bus)
            _write_joint_positions(follower_bus, action)
        else:
            action = obs_state.copy()

        dataset.add_frame({
            "observation.images.cam0": frame0_rgb,
            "observation.images.cam1": frame1_rgb,
            "observation.state":       obs_state,
            "action":                  action,
            "task":                    task,
        })

        frame_count += 1

        if show_preview:
            draw_preview(frame0_bgr, frame1_bgr, "Recording",
                         episode_idx, total_episodes, frame_count, task)

        elapsed = time.perf_counter() - t0
        precise_sleep(max(dt - elapsed, 0.0))

    return frame_count


def reset_phase(
    cam0, cam1, follower_bus, leader_bus, fps, events,
    episode_idx=0, total_episodes=0, task="", show_preview=False,
):
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
            frame0_bgr = cam0.read_bgr()
            frame1_bgr = cam1.read_bgr()
            draw_preview(frame0_bgr, frame1_bgr,
                         "Reset — Press -> to continue",
                         episode_idx, total_episodes, 0, task)
        else:
            precise_sleep(dt)


# ---------------------------------------------------------------------------
# Interactive dataset selector
# ---------------------------------------------------------------------------

def select_dataset(preselect):
    W = 76
    print()
    print("╔" + "═" * W + "╗")
    print("║  Dual USB Camera LeRobot Dataset Recorder" + " " * (W - 42) + "║")
    print("║" + f"  Task prompt (both datasets): \"{TASK_PROMPT}\"".ljust(W) + "║")
    print("╠" + "═" * W + "╣")

    for key, ds in DATASETS.items():
        print("║" + " " * W + "║")
        print("║" + f"  [{key}]  {ds['label']}".ljust(W) + "║")
        print("║" + " " * W + "║")
        for line in ds["description"].splitlines():
            print("║" + ("      " + line).ljust(W) + "║")
        print("║" + " " * W + "║")
        print("║" + f"      Saved to:  datasets/{ds['folder']}/".ljust(W) + "║")
        print("║" + f"      Target:    {ds['num_episodes']} episodes".ljust(W) + "║")
        if key != list(DATASETS.keys())[-1]:
            print("╠" + "═" * W + "╣")

    print("╚" + "═" * W + "╝")
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
# modality.json writer (required by GR00T training)
# ---------------------------------------------------------------------------

def _write_modality_json(root: Path) -> None:
    """Write meta/modality.json mapping dataset keys to GR00T modality names."""
    import json
    modality = {
        "state": {
            "single_arm": {"start": 0, "end": 5},
            "gripper":    {"start": 5, "end": 6},
        },
        "action": {
            "single_arm": {"start": 0, "end": 5},
            "gripper":    {"start": 5, "end": 6},
        },
        "video": {
            "cam0": {"original_key": "observation.images.cam0"},
            "cam1": {"original_key": "observation.images.cam1"},
        },
        "annotation": {
            "human.task_description": {"original_key": "task_index"},
        },
    }
    out = root / "meta" / "modality.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(modality, f, indent=4)
    logging.info("Wrote %s", out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Record a LeRobot dataset from two USB cameras."
    )

    # Dataset selection
    parser.add_argument("--dataset", default=None, metavar="N")
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--episode-time", type=float, default=30.0, dest="episode_time_s")
    parser.add_argument("--vcodec", default="h264_nvenc")
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--private", action="store_true")

    # Cameras
    parser.add_argument("--cam0", type=int, default=0,
                        help="OpenCV device index for camera 0 (default: 0 → /dev/video0)")
    parser.add_argument("--cam1", type=int, default=2,
                        help="OpenCV device index for camera 1 (default: 2 → /dev/video2)")
    parser.add_argument("--cam-width",  type=int, default=1920,
                        help="Requested camera width (default: 1920)")
    parser.add_argument("--cam-height", type=int, default=1080,
                        help="Requested camera height (default: 1080)")

    # Follower arm
    parser.add_argument("--follower-port", default="/dev/ttyACM0")
    parser.add_argument("--follower-id",   default="my_awesome_follower_arm")
    parser.add_argument("--calibration-dir", default=DEFAULT_FOLLOWER_CALIBRATION_DIR)
    parser.add_argument("--leader-calibration-dir", default=DEFAULT_LEADER_CALIBRATION_DIR)

    # Leader arm (teleop) — optional
    parser.add_argument("--leader-port", default=None)
    parser.add_argument("--leader-id",   default="my_leader")

    # Misc
    parser.add_argument("--log-level",  type=int, default=20)
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--no-preview", action="store_true")

    args = parser.parse_args()

    init_logging()
    logging.getLogger().setLevel(args.log_level)

    # ------------------------------------------------------------------ Dataset selection
    ds_config = select_dataset(args.dataset)

    repo_id = args.repo_id or ds_config["repo_id"]
    root    = Path(args.root) if args.root else DATASETS_ROOT / ds_config["folder"]
    task    = TASK_PROMPT

    # ------------------------------------------------------------------ Folder check
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
        existing_episodes = _count_existing_episodes(root)
        default_total = max(ds_config["num_episodes"], existing_episodes + 1)
        print()
        print(f"  {existing_episodes} episode(s) already recorded.")
        while True:
            raw = input(f"  Record up to how many episodes in total? [default: {default_total}]: ").strip()
            if raw == "":
                num_episodes = default_total; break
            if raw.isdigit() and int(raw) > existing_episodes:
                num_episodes = int(raw); break
            print("  Must be greater than existing episode count or leave blank.")
        print(f"  Will record {num_episodes - existing_episodes} more episode(s).\n")
    else:
        default_eps = ds_config["num_episodes"]
        print()
        while True:
            raw = input(f"  How many episodes do you want to record? [default: {default_eps}]: ").strip()
            if raw == "":
                num_episodes = default_eps; break
            if raw.isdigit() and int(raw) > 0:
                num_episodes = int(raw); break
            print("  Please enter a positive whole number.")
        print()

    # ------------------------------------------------------------------ Cameras
    logging.info("Opening cameras (device %d and %d) …", args.cam0, args.cam1)
    cam0 = USBCamera(args.cam0, args.cam_width, args.cam_height)
    cam1 = USBCamera(args.cam1, args.cam_width, args.cam_height)
    cam0.connect()
    cam1.connect()

    # ------------------------------------------------------------------ Preview
    show_preview = not args.no_preview
    if show_preview:
        try:
            cv2.namedWindow(_PREVIEW_WIN, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
            cv2.waitKey(1)
        except cv2.error as e:
            logging.warning("Preview unavailable (%s). Continuing without preview.", e)
            show_preview = False

    # ------------------------------------------------------------------ Arms
    follower_cal = os.path.join(args.calibration_dir, f"{args.follower_id}.json")
    follower_bus = _create_and_connect_bus(args.follower_port, follower_cal, "follower")

    leader_bus = None
    if args.leader_port:
        leader_cal = os.path.join(args.leader_calibration_dir, f"{args.leader_id}.json")
        leader_bus = _create_and_connect_bus(args.leader_port, leader_cal, "leader", is_leader=True)

    # ------------------------------------------------------------------ Dataset
    features = build_features(cam0.height, cam0.width, cam1.height, cam1.width)

    if args.resume:
        root.mkdir(parents=True, exist_ok=True)
        dataset = LeRobotDataset(
            repo_id, root=str(root),
            streaming_encoding=True, encoder_threads=2,
        )
        dataset.start_image_writer(num_processes=0, num_threads=4)
    else:
        dataset = LeRobotDataset.create(
            repo_id, args.fps, root=str(root),
            robot_type="so101_dual_usb",
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
            recorded = dataset.num_episodes

            while recorded < num_episodes and not events["stop_recording"]:
                log_say(f"Recording episode {recorded + 1} of {num_episodes}", play_sounds=False)
                print(f"\n  Episode {recorded + 1} / {num_episodes}  |  task: \"{task}\"")

                events["exit_early"] = False
                events["rerecord_episode"] = False

                n_frames = record_episode(
                    dataset=dataset,
                    cam0=cam0, cam1=cam1,
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

                if events["rerecord_episode"]:
                    log_say("Discarding episode — re-recording", play_sounds=False)
                    dataset.clear_episode_buffer()
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    continue

                dataset.save_episode()
                recorded += 1
                log_say(f"Episode {recorded} saved.", play_sounds=False)
                print(f"  Saved. Progress: {recorded} / {num_episodes}")

                if not events["stop_recording"] and recorded < num_episodes:
                    log_say("Reset the environment", play_sounds=False)
                    events["exit_early"] = False
                    reset_phase(
                        cam0=cam0, cam1=cam1,
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

        cam0.disconnect()
        cam1.disconnect()

        try:
            if follower_bus.is_connected:
                follower_bus.disconnect(disable_torque=True)
        except Exception as e:
            logging.warning("Could not disconnect follower: %s", e)

        try:
            if leader_bus is not None and leader_bus.is_connected:
                leader_bus.disconnect(disable_torque=False)
        except Exception as e:
            logging.warning("Could not disconnect leader: %s", e)

        if listener:
            listener.stop()

        dataset.finalize()
        logging.info("Dataset finalised at %s", dataset.root)

        # Write modality.json so GR00T training can find the camera keys
        _write_modality_json(Path(dataset.root))

        if not args.no_push:
            logging.info("Pushing dataset to Hugging Face Hub …")
            dataset.push_to_hub(private=args.private)
        else:
            logging.info("Skipping Hub push (--no-push).")


if __name__ == "__main__":
    main()
