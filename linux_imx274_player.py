# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# See README.md for detailed information.

import argparse
import csv
import ctypes
import json
import logging
import os
import time

import cuda.bindings.driver as cuda
import holoscan

import hololink as hololink_module

import cupy as cp
import numpy as np


# ---------------------------------------------------------------------------
# SO-101 motor layout
# ---------------------------------------------------------------------------

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

DEFAULT_CALIBRATION_PATH = (
    "/home/latticeapp/.cache/huggingface/lerobot/calibration"
    "/robots/so_follower/{robot_id}.json"
)


def _create_motor_bus(port, calibration_path):
    """Build a FeetechMotorsBus with the SO-101 motor layout."""
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    norm_body = MotorNormMode.DEGREES
    motors = {
        "shoulder_pan": Motor(1, "sts3215", norm_body),
        "shoulder_lift": Motor(2, "sts3215", norm_body),
        "elbow_flex": Motor(3, "sts3215", norm_body),
        "wrist_flex": Motor(4, "sts3215", norm_body),
        "wrist_roll": Motor(5, "sts3215", norm_body),
        "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    }

    calibration = None
    if calibration_path and os.path.exists(calibration_path):
        with open(calibration_path) as f:
            raw = json.load(f)
        calibration = {
            name: MotorCalibration(**data) for name, data in raw.items()
        }
        logging.info("Loaded calibration from %s", calibration_path)
    else:
        logging.warning("No calibration file at %s", calibration_path)

    return FeetechMotorsBus(port=port, motors=motors, calibration=calibration)


def _connect_and_configure_bus(bus):
    """Open the serial port, write calibration, and set PID gains."""
    from lerobot.motors.feetech import OperatingMode

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
    logging.info("Motor bus connected and configured on %s", bus.port)


# ---------------------------------------------------------------------------
# PolicyClientOp — reads state, converts frame, queries policy server
# ---------------------------------------------------------------------------

class PolicyClientOp(holoscan.core.Operator):
    """Receives the demosaic GPU tensor, reads joint state from the shared
    motor bus, and queries the external GR00T policy server over ZMQ.

    Runs inference every ``action_horizon`` ticks and caches the returned
    16-step action chunk.  On each tick it emits the next action step as a
    Python dict for RobotActionOp to execute."""

    DEFAULT_GAIN = 0.7

    def __init__(self, fragment, name, bus, policy_host, policy_port,
                 action_horizon, lang_instruction, gain=None):
        super().__init__(fragment, name)
        self._bus = bus
        self._policy_host = policy_host
        self._policy_port = policy_port
        self._action_horizon = action_horizon
        self._lang = lang_instruction
        gain = gain if gain is not None else self.DEFAULT_GAIN
        self._scale = cp.float32(255.0 / 4095.0 * gain)
        self._client = None
        self._action_chunk = None
        self._step = 0

    def setup(self, spec):
        spec.input("image")
        spec.output("action")

    def start(self):
        from gr00t.policy.server_client import PolicyClient

        self._client = PolicyClient(
            host=self._policy_host,
            port=self._policy_port,
            strict=False,
        )
        logging.info(
            "PolicyClientOp: connected to %s:%d",
            self._policy_host, self._policy_port,
        )

    def compute(self, op_input, op_output, context):
        message = op_input.receive("image")

        if isinstance(message, dict):
            image_tensor = message.get("") or list(message.values())[0]
        else:
            image_tensor = message

        state_dict = self._bus.sync_read("Present_Position")
        state = np.array(
            [state_dict[name] for name in JOINT_NAMES], dtype=np.float32,
        )

        if self._action_chunk is None or self._step >= self._action_horizon:
            frame = cp.asarray(image_tensor)
            if frame.ndim == 3 and frame.shape[-1] == 4:
                frame = frame[:, :, :3]
            if frame.dtype == cp.uint16:
                frame = (frame * self._scale).clip(0, 255).astype(cp.uint8)
            elif cp.issubdtype(frame.dtype, cp.floating):
                frame = (cp.clip(frame, 0.0, 1.0) * 255).astype(cp.uint8)
            frame = cp.ascontiguousarray(frame)
            rgb = cp.asnumpy(frame)

            obs = {
                "video": {
                    "front": rgb[np.newaxis, np.newaxis, ...],
                },
                "state": {
                    "single_arm": state[:5][np.newaxis, np.newaxis, ...],
                    "gripper": state[5:6][np.newaxis, np.newaxis, ...],
                },
                "language": {
                    "annotation.human.task_description": [[self._lang]],
                },
            }

            t0 = time.time()
            action_chunk, _info = self._client.get_action(obs)
            dt_ms = (time.time() - t0) * 1000
            logging.info("PolicyClientOp: inference %.0f ms", dt_ms)

            self._action_chunk = action_chunk
            self._step = 0

        arm = self._action_chunk["single_arm"][0][self._step]
        grip = self._action_chunk["gripper"][0][self._step]
        full = np.concatenate([arm, grip], axis=0)

        action_dict = {
            f"{JOINT_NAMES[i]}.pos": float(full[i])
            for i in range(len(JOINT_NAMES))
        }
        action_dict["_inference_step"] = self._step
        self._step += 1

        op_output.emit(action_dict, "action")


# ---------------------------------------------------------------------------
# RobotActionOp — writes motor commands to the SO-101 arm
# ---------------------------------------------------------------------------

class RobotActionOp(holoscan.core.Operator):
    """Receives an action dict and writes goal positions to the motor bus.

    Two optional CSV logs:
    - ``log_path``:      policy predictions (goal positions) written BEFORE sync_write.
    - ``sent_log_path``: actual sent commands + present positions read AFTER sync_write,
                         so you can compare commanded vs actual joint positions.
    """

    _CSV_FIELDS = ["timestamp", "episode_step", "inference_step"] + JOINT_NAMES
    _SENT_FIELDS = (
        ["timestamp", "episode_step"]
        + [f"sent_{j}" for j in JOINT_NAMES]
        + [f"actual_{j}" for j in JOINT_NAMES]
    )

    def __init__(self, fragment, name, bus, log_path=None, sent_log_path=None):
        super().__init__(fragment, name)
        self._bus = bus
        self._log_path = log_path
        self._sent_log_path = sent_log_path
        self._csv_file = None
        self._csv_writer = None
        self._sent_file = None
        self._sent_writer = None
        self._episode_step = 0

    def setup(self, spec):
        spec.input("action")

    def start(self):
        if self._log_path:
            self._csv_file = open(self._log_path, "w", newline="")
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=self._CSV_FIELDS
            )
            self._csv_writer.writeheader()
            self._csv_file.flush()
            logging.info("RobotActionOp: logging predictions to %s", self._log_path)

        if self._sent_log_path:
            self._sent_file = open(self._sent_log_path, "w", newline="")
            self._sent_writer = csv.DictWriter(
                self._sent_file, fieldnames=self._SENT_FIELDS
            )
            self._sent_writer.writeheader()
            self._sent_file.flush()
            logging.info("RobotActionOp: logging sent commands to %s", self._sent_log_path)

    def stop(self):
        if self._csv_file:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
            logging.info(
                "RobotActionOp: closed prediction log after %d steps", self._episode_step
            )
        if self._sent_file:
            self._sent_file.flush()
            self._sent_file.close()
            self._sent_file = None
            self._sent_writer = None
            logging.info("RobotActionOp: closed sent-commands log.")

    def compute(self, op_input, op_output, context):
        action_dict = op_input.receive("action")
        goal_pos = {
            key.removesuffix(".pos"): val
            for key, val in action_dict.items()
            if key.endswith(".pos")
        }

        # --- Log VLA predictions (before send) ---
        if self._csv_writer is not None:
            inference_step = action_dict.get("_inference_step", "")
            row = {
                "timestamp": time.time(),
                "episode_step": self._episode_step,
                "inference_step": inference_step,
            }
            for joint in JOINT_NAMES:
                row[joint] = round(goal_pos.get(joint, float("nan")), 4)
            self._csv_writer.writerow(row)

        self._episode_step += 1
        self._bus.sync_write("Goal_Position", goal_pos)

        # --- Log sent commands + actual positions (after send) ---
        if self._sent_writer is not None:
            try:
                present = self._bus.sync_read("Present_Position")
            except Exception:
                present = {j: float("nan") for j in JOINT_NAMES}
            row = {
                "timestamp": time.time(),
                "episode_step": self._episode_step,
            }
            for joint in JOINT_NAMES:
                row[f"sent_{joint}"] = round(goal_pos.get(joint, float("nan")), 4)
                row[f"actual_{joint}"] = round(present.get(joint, float("nan")), 4)
            self._sent_writer.writerow(row)


# ---------------------------------------------------------------------------
# Holoscan Application
# ---------------------------------------------------------------------------

class HoloscanApplication(holoscan.core.Application):
    def __init__(
        self,
        headless,
        fullscreen,
        cuda_context,
        cuda_device_ordinal,
        hololink_channel,
        camera,
        camera_mode,
        frame_limit,
        motor_bus=None,
        policy_host=None,
        policy_port=5555,
        action_horizon=8,
        lang_instruction="Move the blue dice",
        action_log=None,
        sent_log=None,
    ):
        logging.info("__init__")
        super().__init__()
        self._headless = headless
        self._fullscreen = fullscreen
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._hololink_channel = hololink_channel
        self._camera = camera
        self._camera_mode = camera_mode
        self._frame_limit = frame_limit
        self._motor_bus = motor_bus
        self._policy_host = policy_host
        self._policy_port = policy_port
        self._action_horizon = action_horizon
        self._lang_instruction = lang_instruction
        self._action_log = action_log
        self._sent_log = sent_log

    def compose(self):
        logging.info("compose")
        if self._frame_limit:
            self._count = holoscan.conditions.CountCondition(
                self,
                name="count",
                count=self._frame_limit,
            )
            condition = self._count
        else:
            self._ok = holoscan.conditions.BooleanCondition(
                self, name="ok", enable_tick=True
            )
            condition = self._ok
        self._camera.set_mode(self._camera_mode)

        csi_to_bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="pool",
            # storage_type of 1 is device memory
            storage_type=1,
            block_size=self._camera._width
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera._height,
            num_blocks=2,
        )
        csi_to_bayer_operator = hololink_module.operators.CsiToBayerOp(
            self,
            name="csi_to_bayer",
            allocator=csi_to_bayer_pool,
            cuda_device_ordinal=self._cuda_device_ordinal,
        )
        self._camera.configure_converter(csi_to_bayer_operator)

        frame_size = csi_to_bayer_operator.get_csi_length()
        frame_context = self._cuda_context
        receiver_operator = hololink_module.operators.LinuxReceiverOperator(
            self,
            condition,
            name="receiver",
            frame_size=frame_size,
            frame_context=frame_context,
            hololink_channel=self._hololink_channel,
            device=self._camera,
        )

        pixel_format = self._camera.pixel_format()
        bayer_format = self._camera.bayer_format()
        image_processor_operator = hololink_module.operators.ImageProcessorOp(
            self,
            name="image_processor",
            # Optical black value for imx274 is 50
            optical_black=50,
            bayer_format=bayer_format.value,
            pixel_format=pixel_format.value,
        )

        rgba_components_per_pixel = 4
        bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="pool",
            # storage_type of 1 is device memory
            storage_type=1,
            block_size=self._camera._width
            * rgba_components_per_pixel
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera._height,
            num_blocks=2,
        )
        demosaic = holoscan.operators.BayerDemosaicOp(
            self,
            name="demosaic",
            pool=bayer_pool,
            generate_alpha=True,
            alpha_value=65535,
            bayer_grid_pos=bayer_format.value,
            interpolation_mode=0,
        )

        visualizer = holoscan.operators.HolovizOp(
            self,
            name="holoviz",
            fullscreen=self._fullscreen,
            headless=self._headless,
            framebuffer_srgb=True,
        )

        # Camera pipeline: receiver → csi_to_bayer → image_processor → demosaic
        self.add_flow(receiver_operator, csi_to_bayer_operator, {("output", "input")})
        self.add_flow(
            csi_to_bayer_operator, image_processor_operator, {("output", "input")}
        )
        self.add_flow(image_processor_operator, demosaic, {("output", "receiver")})
        self.add_flow(demosaic, visualizer, {("transmitter", "receivers")})

        # Robot control branch (only when a motor bus is provided)
        if self._motor_bus is not None:
            policy_client_op = PolicyClientOp(
                self,
                name="policy_client",
                bus=self._motor_bus,
                policy_host=self._policy_host,
                policy_port=self._policy_port,
                action_horizon=self._action_horizon,
                lang_instruction=self._lang_instruction,
            )
            robot_action_op = RobotActionOp(
                self,
                name="robot_action",
                bus=self._motor_bus,
                log_path=self._action_log,
                sent_log_path=self._sent_log,
            )
            self.add_flow(demosaic, policy_client_op, {("transmitter", "image")})
            self.add_flow(policy_client_op, robot_action_op, {("action", "action")})


def main():
    parser = argparse.ArgumentParser()
    modes = hololink_module.sensors.imx274.imx274_mode.Imx274_Mode
    mode_choices = [mode.value for mode in modes]
    mode_help = " ".join([f"{mode.value}:{mode.name}" for mode in modes])
    parser.add_argument(
        "--camera-mode",
        type=int,
        choices=mode_choices,
        default=mode_choices[0],
        help=mode_help,
    )
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument(
        "--fullscreen", action="store_true", help="Run in fullscreen mode"
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help="Exit after receiving this many frames",
    )
    default_configuration = os.path.join(
        os.path.dirname(__file__), "example_configuration.yaml"
    )
    parser.add_argument(
        "--configuration",
        default=default_configuration,
        help="Configuration file",
    )
    parser.add_argument(
        "--hololink",
        default="192.168.0.2",
        help="IP address of Hololink board",
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level to display",
    )
    parser.add_argument(
        "--expander-configuration",
        type=int,
        default=0,
        choices=(0, 1),
        help="I2C Expander configuration",
    )
    parser.add_argument(
        "--pattern",
        type=int,
        choices=range(12),
        help="Configure to display a test pattern.",
    )

    # Robot control arguments
    parser.add_argument(
        "--robot-port",
        default="/dev/ttyACM0",
        help="Serial port for the SO-101 follower arm",
    )
    parser.add_argument(
        "--robot-id",
        default="my_awesome_follower_arm",
        help="Robot calibration ID (matches calibration filename)",
    )
    parser.add_argument(
        "--policy-host",
        default="localhost",
        help="GR00T policy server hostname",
    )
    parser.add_argument(
        "--policy-port",
        type=int,
        default=5555,
        help="GR00T policy server port",
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=8,
        help="Number of action steps to execute per inference call",
    )
    parser.add_argument(
        "--lang-instruction",
        default="Move the blue dice",
        help="Language instruction for the GR00T policy",
    )
    parser.add_argument(
        "--no-robot",
        action="store_true",
        help="Disable robot control (camera-only mode)",
    )
    parser.add_argument(
        "--action-log",
        default=None,
        metavar="PATH",
        help="Write predicted joint goal positions to a CSV file (e.g. /tmp/actions.csv)",
    )
    parser.add_argument(
        "--sent-log",
        default=None,
        metavar="PATH",
        help="Write actually-sent motor commands + present positions to a CSV file",
    )

    args = parser.parse_args()
    hololink_module.logging_level(args.log_level)
    logging.info("Initializing.")

    # Get a handle to the GPU
    (cu_result,) = cuda.cuInit(0)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_device_ordinal = 0
    cu_result, cu_device = cuda.cuDeviceGet(cu_device_ordinal)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_result, cu_context = cuda.cuDevicePrimaryCtxRetain(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS

    # Get a handle to the Hololink device
    channel_metadata = hololink_module.Enumerator.find_channel(channel_ip=args.hololink)
    hololink_channel = hololink_module.DataChannel(channel_metadata)

    # Get a handle to the camera
    camera = hololink_module.sensors.imx274.dual_imx274.Imx274Cam(
        hololink_channel, expander_configuration=args.expander_configuration
    )
    camera_mode = hololink_module.sensors.imx274.imx274_mode.Imx274_Mode(
        args.camera_mode
    )

    # Set up the motor bus (if robot control is enabled)
    motor_bus = None
    if not args.no_robot:
        calibration_path = DEFAULT_CALIBRATION_PATH.format(robot_id=args.robot_id)
        motor_bus = _create_motor_bus(args.robot_port, calibration_path)
        _connect_and_configure_bus(motor_bus)

    # Set up the application
    application = HoloscanApplication(
        args.headless,
        args.fullscreen,
        cu_context,
        cu_device_ordinal,
        hololink_channel,
        camera,
        camera_mode,
        args.frame_limit,
        motor_bus=motor_bus,
        policy_host=args.policy_host,
        policy_port=args.policy_port,
        action_horizon=args.action_horizon,
        lang_instruction=args.lang_instruction,
        action_log=args.action_log,
        sent_log=args.sent_log,
    )
    application.config(args.configuration)

    # Run it.
    hololink = hololink_channel.hololink()
    hololink.start()
    try:
        hololink.reset()
        camera.setup_clock()
        camera.configure(camera_mode)
        camera.set_digital_gain_reg(0x4)
        if args.pattern is not None:
            camera.test_pattern(args.pattern)
        application.run()
    finally:
        hololink.stop()
        if motor_bus is not None:
            motor_bus.disconnect(disable_torque=True)
            logging.info("Motor bus disconnected.")

    (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS


if __name__ == "__main__":
    main()
