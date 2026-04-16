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
#
# -----------------------------------------------------------------------------
# Legacy / comparison build: Holoviz + v4l2loopback fake camera (YUYV)
# -----------------------------------------------------------------------------
# Use this script to compare what Holoviz displays (RGBA uint16 + sRGB framebuffer)
# against what applications see when they open the loopback device (e.g. ffplay,
# OpenCV) — the V4L2 path converts RGB -> YUYV422 and can look different.
#
# Requires (for V4L2 branch): v4l2loopback on the host, /dev/videoN visible in the container,
# pyfakewebcam, opencv-python-headless. Use --no-v4l2-sink for Holoviz-only if loopback is missing.
# Example (host, match --camera-mode):  mode 1 -> -video_size 1920x1080 ; mode 0/2 -> 3840x2160
#   ffplay -f v4l2 -input_format yuyv422 -video_size 1920x1080 /dev/video10
# Do not use this for the integrated GR00T pipeline; use linux_imx274_player.py.
# -----------------------------------------------------------------------------

import argparse
import ctypes
import logging
import os

import cuda.bindings.driver as cuda
import holoscan

import hololink as hololink_module

import cv2
import cupy as cp
import numpy as np
import pyfakewebcam


def _v4l2_missing_help(device: str) -> str:
    return (
        f"v4l2loopback device not found: {device}\n"
        "On the Jetson host (not inside Docker), create the device first, e.g.:\n"
        "  sudo modprobe v4l2loopback devices=1 video_nr=10 card_label=HolovizBridge\n"
        "  ls -l /dev/video*\n"
        "If the node is not /dev/video10, pass: --v4l2-device /dev/video<N>\n"
        "Then restart the container (or rely on -v /dev:/dev in docker/demo.sh).\n"
        "To run Holoviz only without a loopback device, use: --no-v4l2-sink"
    )


_LINEAR_GAIN = np.float32(255.0 / 4095.0 * 0.7)  # original linear scale


def _linear_to_srgb_gpu(linear: "cp.ndarray") -> "cp.ndarray":
    """Apply the sRGB piecewise transfer function to a CuPy float32 array in [0, 1].

    Mirrors the hardware sRGB conversion that HolovizOp applies during scanout.
    IMX274 is a 12-bit sensor; callers should normalize with / 4095.0 before calling.
    """
    linear = cp.clip(linear, 0.0, 1.0)
    return cp.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * cp.power(cp.maximum(linear, 0.0031308), 1.0 / 2.4) - 0.055,
    )


class V4L2SinkOp(holoscan.core.Operator):
    """Streams demosaic output to a v4l2loopback device as YUYV.

    Pass ``use_linear=True`` to use the old linear+gain conversion so you can
    compare it against the sRGB output side-by-side.
    """

    def __init__(self, fragment, name, device="/dev/video10",
                 width=1920, height=1080, use_linear=False, exposure=1.0,
                 rgb_pipe=None):
        super().__init__(fragment, name)
        self.width = width
        self.height = height
        self._use_linear = use_linear
        self._exposure = cp.float32(exposure)
        self._rgb_pipe = None
        # pyfakewebcam handles V4L2 device setup (ioctl VIDIOC_S_FMT);
        # we bypass its schedule_frame which uses the removed ndarray.tostring().
        if not os.path.exists(device):
            raise FileNotFoundError(_v4l2_missing_help(device))
        try:
            self._cam = pyfakewebcam.FakeWebcam(device, width, height)
        except FileNotFoundError as e:
            raise FileNotFoundError(_v4l2_missing_help(device)) from e
        self._fd = self._cam._video_device
        self._expected_bytes = width * height * 2  # YUYV = 2 bytes/pixel
        self._yuyv_buf = np.empty((height, width * 2), dtype=np.uint8)
        if rgb_pipe:
            if not os.path.exists(rgb_pipe):
                os.mkfifo(rgb_pipe)
            # Open non-blocking so we don't hang if ffplay isn't connected yet
            pipe_fd = os.open(rgb_pipe, os.O_WRONLY | os.O_NONBLOCK)
            self._rgb_pipe = pipe_fd
            logging.info("V4L2SinkOp: streaming RGB to pipe %s", rgb_pipe)
            logging.info(
                "  On host run: ffplay -f rawvideo -pixel_format rgb24 "
                "-video_size %dx%d %s", width, height, rgb_pipe,
            )
        logging.info(
            "V4L2SinkOp: using %s conversion",
            "LINEAR (old)" if use_linear else "sRGB (new)",
        )

    def setup(self, spec):
        spec.input("input")

    def compute(self, op_input, op_output, context):
        message = op_input.receive("input")

        if isinstance(message, dict):
            image_tensor = message.get("") or list(message.values())[0]
        else:
            image_tensor = message

        frame = cp.asarray(image_tensor)

        if frame.ndim == 3 and frame.shape[-1] == 4:
            frame = frame[:, :, :3]

        if frame.dtype == cp.uint16:
            if self._use_linear:
                frame = (frame * cp.float32(_LINEAR_GAIN)).clip(0, 255).astype(cp.uint8)
            else:
                linear = (frame.astype(cp.float32) / 4095.0) * self._exposure
                frame = (_linear_to_srgb_gpu(linear) * 255.0).clip(0, 255).astype(cp.uint8)
        elif cp.issubdtype(frame.dtype, cp.floating):
            if self._use_linear:
                frame = (cp.clip(frame, 0.0, 1.0) * 255).astype(cp.uint8)
            else:
                frame = (_linear_to_srgb_gpu(frame * self._exposure) * 255.0).clip(0, 255).astype(cp.uint8)

        frame = cp.ascontiguousarray(frame)
        rgb = cp.asnumpy(frame)

        if self._rgb_pipe is not None:
            try:
                os.write(self._rgb_pipe, rgb.tobytes())
            except BlockingIOError:
                pass  # ffplay not yet connected or too slow — drop the frame

        yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV)
        y = yuv[:, :, 0]
        u = yuv[:, :, 1]
        v = yuv[:, :, 2]
        buf = self._yuyv_buf
        buf[:, 0::4] = y[:, 0::2]
        buf[:, 1::4] = (u[:, 0::2] >> 1) + (u[:, 1::2] >> 1)
        buf[:, 2::4] = y[:, 1::2]
        buf[:, 3::4] = (v[:, 0::2] >> 1) + (v[:, 1::2] >> 1)

        try:
            os.write(self._fd, buf.tobytes()[:self._expected_bytes])
        except OSError as e:
            logging.error("V4L2 write failed: %s", e)


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
        v4l2_device,
        use_v4l2_sink,
        use_linear=False,
        exposure=1.0,
        rgb_pipe=None,
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
        self._v4l2_device = v4l2_device
        self._use_v4l2_sink = use_v4l2_sink
        self._use_linear = use_linear
        self._exposure = exposure
        self._rgb_pipe = rgb_pipe

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
            optical_black=50,
            bayer_format=bayer_format.value,
            pixel_format=pixel_format.value,
        )

        rgba_components_per_pixel = 4
        bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="pool",
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

        self.add_flow(receiver_operator, csi_to_bayer_operator, {("output", "input")})
        self.add_flow(
            csi_to_bayer_operator, image_processor_operator, {("output", "input")}
        )
        self.add_flow(image_processor_operator, demosaic, {("output", "receiver")})
        self.add_flow(demosaic, visualizer, {("transmitter", "receivers")})
        if self._use_v4l2_sink:
            v4l2_sink = V4L2SinkOp(
                self,
                name="v4l2_sink",
                device=self._v4l2_device,
                width=self._camera._width,
                height=self._camera._height,
                use_linear=self._use_linear,
                exposure=self._exposure,
                rgb_pipe=self._rgb_pipe,
            )
            self.add_flow(demosaic, v4l2_sink, {("transmitter", "input")})


def main():
    parser = argparse.ArgumentParser(
        description="IMX274 player with Holoviz + v4l2loopback sink (comparison build).",
    )
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
    parser.add_argument(
        "--v4l2-device",
        default="/dev/video10",
        help="v4l2loopback device path for fake webcam output",
    )
    parser.add_argument(
        "--no-v4l2-sink",
        action="store_true",
        help="Holoviz only; skip v4l2loopback (use when /dev/videoN is not set up)",
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Use old linear+gain conversion instead of sRGB (for before/after comparison)",
    )
    parser.add_argument(
        "--exposure",
        type=float,
        default=0.3,
        help="Linear exposure multiplier applied before sRGB curve (default 0.3)",
    )
    parser.add_argument(
        "--rgb-pipe",
        default=None,
        metavar="PATH",
        help=(
            "Write raw RGB24 frames to this named pipe for viewing with ffplay. "
            "The pipe is created automatically if it does not exist. "
            "Then on the host run: ffplay -f rawvideo -pixel_format rgb24 "
            "-video_size WxH PATH"
        ),
    )
    args = parser.parse_args()
    hololink_module.logging_level(args.log_level)
    logging.info("Initializing (V4L2 comparison build).")

    (cu_result,) = cuda.cuInit(0)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_device_ordinal = 0
    cu_result, cu_device = cuda.cuDeviceGet(cu_device_ordinal)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_result, cu_context = cuda.cuDevicePrimaryCtxRetain(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS

    channel_metadata = hololink_module.Enumerator.find_channel(channel_ip=args.hololink)
    hololink_channel = hololink_module.DataChannel(channel_metadata)

    camera = hololink_module.sensors.imx274.dual_imx274.Imx274Cam(
        hololink_channel, expander_configuration=args.expander_configuration
    )
    camera_mode = hololink_module.sensors.imx274.imx274_mode.Imx274_Mode(
        args.camera_mode
    )

    use_v4l2 = not args.no_v4l2_sink
    if use_v4l2 and not os.path.exists(args.v4l2_device):
        logging.error("%s", _v4l2_missing_help(args.v4l2_device))
        raise SystemExit(1)

    application = HoloscanApplication(
        args.headless,
        args.fullscreen,
        cu_context,
        cu_device_ordinal,
        hololink_channel,
        camera,
        camera_mode,
        args.frame_limit,
        v4l2_device=args.v4l2_device,
        use_v4l2_sink=use_v4l2,
        use_linear=args.linear,
        exposure=args.exposure,
        rgb_pipe=args.rgb_pipe,
    )
    application.config(args.configuration)

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

    (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS


if __name__ == "__main__":
    main()
