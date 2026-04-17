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
# After each episode there is a --reset-time second window to reset the scene.

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np

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

def _create_and_connect_bus(port, calibration_path, label="arm"):
    """Create, configure and connect a FeetechMotorsBus for an SO-101 arm."""
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
            bus.write("P_Coefficient", motor, 16)
            bus.write("I_Coefficient", motor, 0)
            bus.write("D_Coefficient", motor, 32)
            if motor == "gripper":
                bus.write("Max_Torque_Limit", motor, 500)
                bus.write("Protection_Current", motor, 250)
                bus.write("Overload_Torque", motor, 25)

    logging.info("%s: connected on %s", label, port)
    return bus


def _read_joint_positions(bus) -> np.ndarray:
    """Synchronously read all 6 joint positions; returns float32 array shape (6,)."""
    pos = bus.sync_read("Present_Position")
    return np.array([pos[j] for j in JOINT_NAMES], dtype=np.float32)


def _write_joint_positions(bus, positions: np.ndarray) -> None:
    """Synchronously write 6 joint goal positions from a float32 array shape (6,)."""
    goal = {j: float(positions[i]) for i, j in enumerate(JOINT_NAMES)}
    bus.sync_write("Goal_Position", goal)


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

        elapsed = time.perf_counter() - t0
        precise_sleep(max(dt - elapsed, 0.0))

    return frame_count


def reset_phase(
    camera: ZMQCamera,
    follower_bus,
    leader_bus,
    fps: int,
    reset_time_s: float,
    events: dict,
) -> None:
    """Wait during environment reset — optionally mirrors leader arm."""
    dt = 1.0 / fps
    start_t = time.perf_counter()

    while (time.perf_counter() - start_t) < reset_time_s:
        if events["exit_early"] or events["stop_recording"]:
            break

        if leader_bus is not None:
            action = _read_joint_positions(leader_bus)
            _write_joint_positions(follower_bus, action)

        precise_sleep(dt)


# ---------------------------------------------------------------------------
# Interactive dataset selector
# ---------------------------------------------------------------------------

def select_dataset(preselect: str | None) -> dict:
    """Print the experiment menu and return the chosen dataset config dict."""
    W = 66

    def rule():  return "╠" + "═" * W + "╣"
    def top():   return "╔" + "═" * W + "╗"
    def bot():   return "╚" + "═" * W + "╝"
    def row(s=""):
        padding = W - len(s)
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
    parser.add_argument("--reset-time", type=float, default=10.0, dest="reset_time_s",
                        help="Seconds to reset the environment between episodes (default: 10)")
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

    args = parser.parse_args()

    init_logging()
    logging.getLogger().setLevel(args.log_level)

    # ------------------------------------------------------------------ Dataset selection
    ds_config = select_dataset(args.dataset)

    repo_id     = args.repo_id     or ds_config["repo_id"]
    num_episodes = args.num_episodes or ds_config["num_episodes"]
    root        = Path(args.root)  if args.root else DATASETS_ROOT / ds_config["folder"]
    task        = TASK_PROMPT

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

    # ------------------------------------------------------------------ Arms
    follower_cal = os.path.join(args.calibration_dir, f"{args.follower_id}.json")
    follower_bus = _create_and_connect_bus(args.follower_port, follower_cal, "follower")

    leader_bus = None
    if args.leader_port:
        leader_cal = os.path.join(args.leader_calibration_dir, f"{args.leader_id}.json")
        leader_bus = _create_and_connect_bus(args.leader_port, leader_cal, "leader")

    # ------------------------------------------------------------------ Dataset
    features = build_features(img_h, img_w)
    root.mkdir(parents=True, exist_ok=True)

    if args.resume:
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
            recorded = 0

            while recorded < num_episodes and not events["stop_recording"]:
                log_say(f"Recording episode {dataset.num_episodes + 1} of {num_episodes}", play_sounds=False)
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
                    logging.info("Reset phase: %.0fs …", args.reset_time_s)
                    events["exit_early"] = False
                    reset_phase(
                        camera=camera,
                        follower_bus=follower_bus,
                        leader_bus=leader_bus,
                        fps=args.fps,
                        reset_time_s=args.reset_time_s,
                        events=events,
                    )

    finally:
        log_say("Stopping recording", play_sounds=False)

        camera.disconnect()

        if follower_bus.is_connected:
            follower_bus.disconnect(disable_torque=True)
            logging.info("Follower arm disconnected.")

        if leader_bus is not None and leader_bus.is_connected:
            leader_bus.disconnect(disable_torque=False)
            logging.info("Leader arm disconnected.")

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
