# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Runs INSIDE the hololink-demo Docker container.
#
# Drives the IMX274 camera through the Holoscan pipeline (CSI → Bayer → demosaic →
# sRGB uint8) and publishes every frame as a base64-encoded JPEG over a ZMQ PUB
# socket so that lerobot on the HOST can consume it with ZMQCamera.
#
# The wire format matches lerobot/cameras/zmq/image_server.py exactly:
#   {"timestamps": {"front": <float>}, "images": {"front": "<base64-jpeg>"}}
#
# Usage (inside container):
#   python imx274_zmq_server.py --camera-mode 1 --headless
#
# On the host, point ZMQCameraConfig to:
#   server_address = "localhost"   (or the Jetson IP if running remotely)
#   port           = 5556          (default below)
#   camera_name    = "front"

import argparse
import base64
import ctypes
import json
import logging
import time

import cv2
import cupy as cp
import holoscan
from cuda.bindings import driver as cuda

import hololink as hololink_module


# ---------------------------------------------------------------------------
# sRGB helpers (copied from linux_imx274_player.py for the preview path)
# ---------------------------------------------------------------------------

def _linear_to_srgb_gpu(linear: "cp.ndarray") -> "cp.ndarray":
    linear = cp.clip(linear, 0.0, 1.0)
    return cp.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * cp.power(cp.maximum(linear, 0.0031308), 1.0 / 2.4) - 0.055,
    )


class SrgbConvertOp(holoscan.core.Operator):
    """RGBA uint16 demosaic tensor → RGB uint8 CuPy array (GPU only, no CPU round-trip)."""

    def __init__(self, fragment, name, exposure=1.0):
        super().__init__(fragment, name)
        self._exposure = cp.float32(exposure)

    def setup(self, spec):
        spec.input("image")
        spec.output("converted")

    def compute(self, op_input, op_output, context):
        message = op_input.receive("image")
        image_tensor = message.get("") if isinstance(message, dict) else message
        if image_tensor is None and isinstance(message, dict):
            image_tensor = list(message.values())[0]

        frame = cp.asarray(image_tensor)

        if frame.ndim == 3 and frame.shape[-1] == 4:
            frame = frame[:, :, :3]

        if frame.dtype == cp.uint16:
            linear = (frame.astype(cp.float32) / 4095.0) * self._exposure
            frame = (_linear_to_srgb_gpu(linear) * 255.0).clip(0, 255).astype(cp.uint8)
        elif cp.issubdtype(frame.dtype, cp.floating):
            frame = (_linear_to_srgb_gpu(frame * self._exposure) * 255.0).clip(0, 255).astype(cp.uint8)

        op_output.emit({"": cp.ascontiguousarray(frame)}, "converted")


# ---------------------------------------------------------------------------
# ZmqPublisherOp — receives RGB uint8, publishes JPEG over ZMQ
# ---------------------------------------------------------------------------

class ZmqPublisherOp(holoscan.core.Operator):
    """Encodes each RGB uint8 frame as a JPEG and publishes it over ZMQ PUB.

    The JSON wire format is identical to lerobot's ImageServer so that
    ZMQCamera on the host can consume the stream without modification:
      {"timestamps": {"front": <float>}, "images": {"front": "<base64-jpeg>"}}
    """

    def __init__(self, fragment, name, zmq_port=5556, camera_name="front", jpeg_quality=90):
        super().__init__(fragment, name)
        self._port = zmq_port
        self._camera_name = camera_name
        self._quality = jpeg_quality
        self._context = None
        self._socket = None

    def setup(self, spec):
        spec.input("image")

    def start(self):
        import zmq

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        # Only keep the latest frame in the send buffer — no pile-up when
        # the subscriber is temporarily slow.
        self._socket.setsockopt(zmq.SNDHWM, 2)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(f"tcp://*:{self._port}")
        logging.info("ZmqPublisherOp: publishing on tcp://*:%d (camera_name=%s)", self._port, self._camera_name)

    def stop(self):
        if self._socket:
            self._socket.close()
            self._socket = None
        if self._context:
            self._context.term()
            self._context = None

    def compute(self, op_input, op_output, context):
        import zmq

        message = op_input.receive("image")
        frame_gpu = message.get("") if isinstance(message, dict) else message
        if frame_gpu is None and isinstance(message, dict):
            frame_gpu = list(message.values())[0]

        # GPU → CPU (single copy, contiguous)
        frame_rgb = cp.asnumpy(cp.ascontiguousarray(cp.asarray(frame_gpu)))

        # cv2.imencode expects BGR; our frame is RGB — swap channels
        frame_bgr = frame_rgb[:, :, ::-1]
        ok, buf = cv2.imencode(
            ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, self._quality]
        )
        if not ok:
            logging.warning("ZmqPublisherOp: JPEG encode failed, skipping frame")
            return

        encoded = base64.b64encode(buf).decode("utf-8")
        payload = json.dumps({
            "timestamps": {self._camera_name: time.time()},
            "images": {self._camera_name: encoded},
        })

        try:
            self._socket.send_string(payload, zmq.NOBLOCK)
        except zmq.Again:
            pass  # Subscriber too slow — drop frame rather than stall the pipeline


# ---------------------------------------------------------------------------
# Holoscan Application
# ---------------------------------------------------------------------------

class Imx274ZmqApp(holoscan.core.Application):
    def __init__(
        self,
        headless,
        cuda_context,
        cuda_device_ordinal,
        hololink_channel,
        camera,
        camera_mode,
        exposure,
        zmq_port,
        camera_name,
        jpeg_quality,
        show_preview,
        frame_limit,
    ):
        super().__init__()
        self._headless = headless
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._hololink_channel = hololink_channel
        self._camera = camera
        self._camera_mode = camera_mode
        self._exposure = exposure
        self._zmq_port = zmq_port
        self._camera_name = camera_name
        self._jpeg_quality = jpeg_quality
        self._show_preview = show_preview
        self._frame_limit = frame_limit

    def compose(self):
        logging.info("compose")

        if self._frame_limit:
            condition = holoscan.conditions.CountCondition(self, name="count", count=self._frame_limit)
        else:
            condition = holoscan.conditions.BooleanCondition(self, name="ok", enable_tick=True)

        self._camera.set_mode(self._camera_mode)

        csi_to_bayer_pool = holoscan.resources.BlockMemoryPool(
            self, name="csi_pool",
            storage_type=1,
            block_size=self._camera._width * ctypes.sizeof(ctypes.c_uint16) * self._camera._height,
            num_blocks=2,
        )
        csi_to_bayer = hololink_module.operators.CsiToBayerOp(
            self, name="csi_to_bayer",
            allocator=csi_to_bayer_pool,
            cuda_device_ordinal=self._cuda_device_ordinal,
        )
        self._camera.configure_converter(csi_to_bayer)

        frame_size = csi_to_bayer.get_csi_length()
        receiver = hololink_module.operators.LinuxReceiverOperator(
            self, condition, name="receiver",
            frame_size=frame_size,
            frame_context=self._cuda_context,
            hololink_channel=self._hololink_channel,
            device=self._camera,
        )

        bayer_format = self._camera.bayer_format()
        pixel_format = self._camera.pixel_format()
        image_processor = hololink_module.operators.ImageProcessorOp(
            self, name="image_processor",
            optical_black=50,
            bayer_format=bayer_format.value,
            pixel_format=pixel_format.value,
        )

        bayer_pool = holoscan.resources.BlockMemoryPool(
            self, name="bayer_pool",
            storage_type=1,
            block_size=self._camera._width * 4 * ctypes.sizeof(ctypes.c_uint16) * self._camera._height,
            num_blocks=2,
        )
        demosaic = holoscan.operators.BayerDemosaicOp(
            self, name="demosaic",
            pool=bayer_pool,
            generate_alpha=True,
            alpha_value=65535,
            bayer_grid_pos=bayer_format.value,
            interpolation_mode=0,
        )

        srgb_convert = SrgbConvertOp(self, name="srgb_convert", exposure=self._exposure)

        zmq_publisher = ZmqPublisherOp(
            self, name="zmq_publisher",
            zmq_port=self._zmq_port,
            camera_name=self._camera_name,
            jpeg_quality=self._jpeg_quality,
        )

        # Core pipeline: receiver → csi_to_bayer → image_processor → demosaic → srgb_convert → zmq_publisher
        self.add_flow(receiver, csi_to_bayer, {("output", "input")})
        self.add_flow(csi_to_bayer, image_processor, {("output", "input")})
        self.add_flow(image_processor, demosaic, {("output", "receiver")})
        self.add_flow(demosaic, srgb_convert, {("transmitter", "image")})
        self.add_flow(srgb_convert, zmq_publisher, {("converted", "image")})

        # Optional on-screen preview (disable with --headless for pure server mode)
        if self._show_preview and not self._headless:
            preview = holoscan.operators.HolovizOp(
                self, name="preview",
                fullscreen=False,
                headless=False,
                framebuffer_srgb=False,
                tensors=[{"name": "", "type": "color", "opacity": 1.0}],
            )
            self.add_flow(srgb_convert, preview, {("converted", "receivers")})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stream IMX274 camera over ZMQ for lerobot dataset recording."
    )
    modes = hololink_module.sensors.imx274.imx274_mode.Imx274_Mode
    mode_choices = [m.value for m in modes]
    mode_help = " ".join([f"{m.value}:{m.name}" for m in modes])

    parser.add_argument("--camera-mode", type=int, choices=mode_choices, default=mode_choices[0], help=mode_help)
    parser.add_argument("--hololink", default="192.168.0.2", help="IP address of Hololink board")
    parser.add_argument("--expander-configuration", type=int, default=0, choices=(0, 1))
    parser.add_argument("--headless", action="store_true", default=True, help="Run headless (default: True)")
    parser.add_argument("--no-headless", dest="headless", action="store_false", help="Show preview window")
    parser.add_argument("--preview", action="store_true", help="Open sRGB preview window (requires --no-headless)")
    parser.add_argument("--zmq-port", type=int, default=5556, help="ZMQ PUB port (default: 5556)")
    parser.add_argument("--camera-name", default="front", help="Camera name used in ZMQ message (default: front)")
    parser.add_argument("--jpeg-quality", type=int, default=90, metavar="[1-100]", help="JPEG quality for ZMQ stream (default: 90)")
    parser.add_argument("--exposure", type=float, default=0.3, help="Linear exposure multiplier before sRGB (default: 0.3)")
    parser.add_argument("--frame-limit", type=int, default=None, help="Stop after N frames (useful for testing)")
    parser.add_argument("--log-level", type=int, default=20, help="Python logging level (default: INFO=20)")
    parser.add_argument("--configuration", default=None, help="Holoscan YAML config file")
    parser.add_argument("--pattern", type=int, choices=range(12), help="Camera test pattern (0–11)")

    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(levelname)s:%(name)s: %(message)s")
    hololink_module.logging_level(args.log_level)

    # CUDA context
    (cu_result,) = cuda.cuInit(0)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_result, cu_device = cuda.cuDeviceGet(0)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_result, cu_context = cuda.cuDevicePrimaryCtxRetain(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS

    # Hololink + camera
    channel_meta = hololink_module.Enumerator.find_channel(channel_ip=args.hololink)
    hololink_channel = hololink_module.DataChannel(channel_meta)
    camera = hololink_module.sensors.imx274.dual_imx274.Imx274Cam(
        hololink_channel, expander_configuration=args.expander_configuration
    )
    camera_mode = hololink_module.sensors.imx274.imx274_mode.Imx274_Mode(args.camera_mode)

    app = Imx274ZmqApp(
        headless=args.headless,
        cuda_context=cu_context,
        cuda_device_ordinal=0,
        hololink_channel=hololink_channel,
        camera=camera,
        camera_mode=camera_mode,
        exposure=args.exposure,
        zmq_port=args.zmq_port,
        camera_name=args.camera_name,
        jpeg_quality=args.jpeg_quality,
        show_preview=args.preview,
        frame_limit=args.frame_limit,
    )

    if args.configuration:
        app.config(args.configuration)

    hololink = hololink_channel.hololink()
    hololink.start()
    try:
        hololink.reset()
        camera.setup_clock()
        camera.configure(camera_mode)
        camera.set_digital_gain_reg(0x4)
        if args.pattern is not None:
            camera.test_pattern(args.pattern)
        logging.info("Starting ZMQ stream on port %d …", args.zmq_port)
        app.run()
    finally:
        hololink.stop()

    (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS


if __name__ == "__main__":
    main()
