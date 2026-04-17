#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Send a custom payload (0xDEADBEEF) from AGX Thor to FPGA
# using a Python UDP transmitter (plain UDP -- works on Thor without ConnectX NIC).
#
# Usage:
#   python3 examples/thor_send_payload.py
#   python3 examples/thor_send_payload.py --hololink 192.168.0.2 --payload 0xCAFEBABE --frame-limit 5 --port 5000
#
# Notes:
#   - Sends raw bytes directly as a plain UDP datagram -- no RoCEv2 headers (no BTH/RETH).
#   - Does NOT require the C++ PayloadGeneratorOp build step.
#   - Does NOT require sudo (no raw socket).

import argparse
import logging
import socket

import numpy as np
import holoscan
import holoscan.resources
import holoscan.schedulers

import hololink as hololink_module


class PayloadGeneratorOp(holoscan.core.Operator):
    """
    Generates a fixed payload as a uint32 numpy array and emits it every tick.
    """

    def __init__(self, fragment, *args, payload_value=0xDEADBEEF, repeat=True, **kwargs):
        self._payload_value = payload_value
        self._repeat = repeat
        self._sent = False
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: holoscan.core.OperatorSpec):
        spec.output("output")

    def compute(self, op_input, op_output, context):
        if not self._repeat and self._sent:
            return

        # 0xDEADBEEF as uint32, little-endian on the wire → bytes: EF BE AD DE
        data = np.array([self._payload_value], dtype=np.uint32)

        logging.info(
            f"PayloadGeneratorOp: emitting 0x{self._payload_value:08X}  "
            f"bytes={data.tobytes().hex()}"
        )

        op_output.emit(data, "output")
        self._sent = True


class PythonUdpTransmitterOp(holoscan.core.Operator):
    """
    Pure-Python UDP transmitter.
    Avoids the C++ UdpTransmitterOp which uses receive<shared_ptr<Tensor>>
    and cannot deserialize tensors emitted from Python operators.
    """

    def __init__(self, fragment, *args, ip="", port=5000, **kwargs):
        self._ip = ip
        self._port = port
        self._sock = None
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: holoscan.core.OperatorSpec):
        spec.input("input")

    def start(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        logging.info(f"PythonUdpTransmitterOp: UDP socket ready → {self._ip}:{self._port}")

    def stop(self):
        if self._sock:
            self._sock.close()
            self._sock = None

    def compute(self, op_input, op_output, context):
        data = op_input.receive("input")
        if data is None:
            return
        buf = data.tobytes() if hasattr(data, "tobytes") else bytes(data)
        sent = self._sock.sendto(buf, (self._ip, self._port))
        logging.info(f"PythonUdpTransmitterOp: sent {sent} bytes to {self._ip}:{self._port}")


class ThorSendPayloadApp(holoscan.core.Application):
    def __init__(self, hololink_ip, fpga_port, payload_value, frame_limit):
        super().__init__()
        self._hololink_ip   = hololink_ip
        self._fpga_port     = fpga_port
        self._payload_value = payload_value
        self._frame_limit   = frame_limit

    def compose(self):
        logging.info("compose")

        if self._frame_limit:
            condition = holoscan.conditions.CountCondition(
                self,
                name="count",
                count=self._frame_limit,
            )
        else:
            condition = holoscan.conditions.PeriodicCondition(
                self,
                name="periodic",
                recess_period=1_000_000_000,  # 1 second in nanoseconds
            )

        payload_generator = PayloadGeneratorOp(
            self,
            condition,
            name="payload_generator",
            payload_value=self._payload_value,
            repeat=True,
        )

        udp_transmitter = PythonUdpTransmitterOp(
            self,
            name="udp_transmitter",
            ip=self._hololink_ip,
            port=self._fpga_port,
        )

        self.add_flow(payload_generator, udp_transmitter, {("output", "input")})

        # EventBasedScheduler: wakes the transmitter immediately when the
        # generator emits, avoiding the GreedyScheduler's premature deadlock
        # detection (which stops the pipeline before the transmitter can run).
        self.scheduler(holoscan.schedulers.EventBasedScheduler(
            self,
            name="event_scheduler",
            worker_thread_number=2,
        ))


def main():
    parser = argparse.ArgumentParser(
        description="Send a custom payload from AGX Thor to FPGA via UDP"
    )
    parser.add_argument(
        "--hololink",
        default="192.168.0.2",
        help="IP address of the HSB board (default: 192.168.0.2)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=4791,
        help="UDP destination port on FPGA (default: 4791)",
    )
    parser.add_argument(
        "--payload",
        type=lambda x: int(x, 0),
        default=0xDEADBEEF,
        help="32-bit payload value to send (default: 0xDEADBEEF)",
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=1,
        help="Number of times to send (default: 1, 0 = continuous at 1 Hz)",
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Python logging level (default: 20=INFO)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    hololink_module.logging_level(args.log_level)

    frame_limit = args.frame_limit if args.frame_limit > 0 else None

    logging.info(f"Sending 0x{args.payload:08X} to {args.hololink}:{args.port}")
    logging.info(f"Frame limit: {frame_limit if frame_limit else 'unlimited'}")

    channel_metadata = hololink_module.Enumerator.find_channel(channel_ip=args.hololink)
    logging.info(f"{channel_metadata=}")
    hololink_channel = hololink_module.DataChannel(channel_metadata)

    app = ThorSendPayloadApp(
        hololink_ip=args.hololink,
        fpga_port=args.port,
        payload_value=args.payload,
        frame_limit=frame_limit,
    )

    hololink = hololink_channel.hololink()
    hololink.start()
    try:
        app.run()
        logging.info("Done.")
    finally:
        hololink.stop()


if __name__ == "__main__":
    main()
