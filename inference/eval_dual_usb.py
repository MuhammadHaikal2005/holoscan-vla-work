"""
Closed-loop policy evaluation — SO-101 follower arm + dual USB cameras.

Architecture:
    This script is the *client*. The GR00T policy server must already be
    running (see inference/run_server.sh) before launching this script.

    Camera frames + joint positions → PolicyClient → action chunk → robot

Hardware:
    Follower arm  : Feetech STS3215, /dev/ttyACM0 (by default)
    cam0          : /dev/video0  (front/top view)
    cam1          : /dev/video2  (side view)

Usage:
    conda activate lerobot2
    python inference/eval_dual_usb.py [--options]

    python inference/eval_dual_usb.py \\
        --follower-port /dev/ttyACM0 \\
        --cam0 0 --cam1 2 \\
        --lang "Pick up the cube" \\
        --action-horizon 8
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[1]
_GROOT_DIR = Path.home() / "Isaac-GR00T"

sys.path.insert(0, str(_GROOT_DIR))

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

DEFAULT_FOLLOWER_CALIBRATION = str(
    _REPO_ROOT / "calibration" / "robots" / "so_follower"
)
DEFAULT_LANG = "Pick up the cube"


# ──────────────────────────────────────────────────────────────────────────────
# Camera helper
# ──────────────────────────────────────────────────────────────────────────────

class UsbCamera:
    """Thin OpenCV wrapper that always returns RGB uint8 frames."""

    def __init__(self, index: int, width: int = 640, height: int = 480, fps: int = 30):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {index} (/dev/video{index*2})")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        log.info("Camera %d: %dx%d @ %d fps", index, width, height, fps)

    def read_rgb(self) -> np.ndarray:
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Camera read failed")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def release(self):
        self.cap.release()


# ──────────────────────────────────────────────────────────────────────────────
# Robot arm helpers (mirrors usb_dual_cam_record.py)
# ──────────────────────────────────────────────────────────────────────────────

def _create_follower_bus(port: str, calibration_dir: str):
    import json
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode

    motors = {
        "shoulder_pan":  Motor(1, "sts3215", MotorNormMode.DEGREES),
        "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
        "elbow_flex":    Motor(3, "sts3215", MotorNormMode.DEGREES),
        "wrist_flex":    Motor(4, "sts3215", MotorNormMode.DEGREES),
        "wrist_roll":    Motor(5, "sts3215", MotorNormMode.DEGREES),
        "gripper":       Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    }

    cal_path = os.path.join(calibration_dir, "my_awesome_follower_arm.json")
    calibration = None
    if os.path.exists(cal_path):
        with open(cal_path) as f:
            raw = json.load(f)
        calibration = {name: MotorCalibration(**data) for name, data in raw.items()}
        log.info("Loaded calibration from %s", cal_path)
    else:
        log.warning("No calibration file at %s — positions may be uncalibrated", cal_path)

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
        bus.write("Max_Torque_Limit", "gripper", 500)
        bus.write("Protection_Current", "gripper", 250)
        bus.write("Overload_Torque", "gripper", 25)

    # Enable torque and hold current position before any action arrives
    current = bus.sync_read("Present_Position")
    bus.sync_write("Goal_Position", current)
    for motor in bus.motors:
        bus.write("Torque_Enable", motor, 1)

    log.info("Follower arm connected on %s, torque ON", port)
    return bus


def _read_joints(bus, retries: int = 3) -> np.ndarray:
    for attempt in range(retries):
        try:
            pos = bus.sync_read("Present_Position")
            return np.array([pos[j] for j in JOINT_NAMES], dtype=np.float32)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(0.005)
            else:
                raise RuntimeError(f"Joint read failed after {retries} attempts: {e}") from e


def _write_joints(bus, positions: np.ndarray) -> None:
    goal = {j: float(positions[i]) for i, j in enumerate(JOINT_NAMES)}
    bus.sync_write("Goal_Position", goal)


# ──────────────────────────────────────────────────────────────────────────────
# GR00T observation packaging
# ──────────────────────────────────────────────────────────────────────────────

def _add_batch_time_dims(obs: dict) -> dict:
    """Add (B=1, T=1) dims to all arrays — required by the GR00T policy server."""
    out = {}
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            out[key] = val[np.newaxis, np.newaxis, ...]  # (1, 1, ...)
        elif isinstance(val, dict):
            out[key] = _add_batch_time_dims(val)
        else:
            out[key] = [[val]]  # scalar → [[scalar]]
    return out


def build_policy_input(
    frame0_rgb: np.ndarray,
    frame1_rgb: np.ndarray,
    joints: np.ndarray,
    lang: str,
) -> dict:
    """
    Package raw sensor data into the dict the GR00T policy server expects.

    Expected format (after adding B/T dims):
        video:
            cam0: (1, 1, H, W, 3)  uint8
            cam1: (1, 1, H, W, 3)  uint8
        state:
            single_arm: (1, 1, 5)  float32  [degrees]
            gripper:    (1, 1, 1)  float32  [0-100]
        language:
            annotation.human.task_description: [[str]]
    """
    obs = {
        "video": {
            "cam0": frame0_rgb,
            "cam1": frame1_rgb,
        },
        "state": {
            "single_arm": joints[:5],
            "gripper":    joints[5:6],
        },
        "language": {
            "annotation.human.task_description": lang,
        },
    }
    return _add_batch_time_dims(obs)


# ──────────────────────────────────────────────────────────────────────────────
# Main eval loop
# ──────────────────────────────────────────────────────────────────────────────

def run_eval(args):
    from gr00t.policy.server_client import PolicyClient

    # ── Connect cameras ────────────────────────────────────────────────────────
    log.info("Opening cameras...")
    cam0 = UsbCamera(args.cam0, width=args.cam_width, height=args.cam_height)
    cam1 = UsbCamera(args.cam1, width=args.cam_width, height=args.cam_height)

    # ── Connect follower arm ───────────────────────────────────────────────────
    log.info("Connecting follower arm on %s ...", args.follower_port)
    bus = _create_follower_bus(args.follower_port, args.calibration_dir)

    # ── Connect to GR00T policy server ────────────────────────────────────────
    log.info("Connecting to policy server at %s:%d ...", args.host, args.port)
    client = PolicyClient(host=args.host, port=args.port)
    log.info('Policy ready. Language instruction: "%s"', args.lang)
    log.info("Action horizon: %d steps  |  Control rate: %d Hz", args.action_horizon, args.hz)
    log.info("Press Ctrl+C to stop.")

    step_period = 1.0 / args.hz
    action_chunk: list[np.ndarray] | None = None
    chunk_idx = 0

    try:
        while True:
            t0 = time.time()

            # ── Read sensors every step ────────────────────────────────────────
            frame0 = cam0.read_rgb()
            frame1 = cam1.read_rgb()
            joints = _read_joints(bus)

            # ── Query policy when chunk is exhausted ──────────────────────────
            if action_chunk is None or chunk_idx >= args.action_horizon:
                policy_input = build_policy_input(frame0, frame1, joints, args.lang)
                raw_chunk, _info = client.get_action(policy_input)

                # raw_chunk["single_arm"]: (1, T, 5), raw_chunk["gripper"]: (1, T, 1)
                T = raw_chunk["single_arm"].shape[1]
                action_chunk = [
                    np.concatenate(
                        [raw_chunk["single_arm"][0][t], raw_chunk["gripper"][0][t]],
                        axis=0,
                    )
                    for t in range(T)
                ]
                chunk_idx = 0

            # ── Execute current action step ────────────────────────────────────
            action = action_chunk[chunk_idx]
            chunk_idx += 1
            _write_joints(bus, action)

            log.debug(
                "step=%d  joints=%s  action=%s",
                chunk_idx,
                np.round(joints, 1),
                np.round(action, 1),
            )

            # ── Sleep to maintain target control rate ──────────────────────────
            elapsed = time.time() - t0
            remaining = step_period - elapsed
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        log.info("Interrupted — disabling torque and releasing hardware.")
    finally:
        for motor in bus.motors:
            bus.write("Torque_Enable", motor, 0)
        cam0.release()
        cam1.release()
        log.info("Done.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Run closed-loop GR00T policy on SO-101 + dual USB cameras."
    )

    # Hardware
    p.add_argument("--follower-port", default="/dev/ttyACM0",
                   help="Serial port for the SO-101 follower arm (default: /dev/ttyACM0)")
    p.add_argument("--calibration-dir", default=DEFAULT_FOLLOWER_CALIBRATION,
                   help="Directory containing follower calibration JSON")
    p.add_argument("--cam0", type=int, default=0,
                   help="OpenCV index for camera 0 (default: 0)")
    p.add_argument("--cam1", type=int, default=2,
                   help="OpenCV index for camera 1 (default: 2)")
    p.add_argument("--cam-width", type=int, default=640)
    p.add_argument("--cam-height", type=int, default=480)

    # Policy
    p.add_argument("--host", default="localhost",
                   help="Policy server host (default: localhost)")
    p.add_argument("--port", type=int, default=5555,
                   help="Policy server port (default: 5555)")
    p.add_argument("--lang", default=DEFAULT_LANG,
                   help=f'Language instruction (default: "{DEFAULT_LANG}")')
    p.add_argument("--action-horizon", type=int, default=8,
                   help="How many steps of the 16-step chunk to execute before querying again "
                        "(default: 8)")

    # Timing
    p.add_argument("--hz", type=int, default=30,
                   help="Control loop frequency in Hz (default: 30)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(args)
