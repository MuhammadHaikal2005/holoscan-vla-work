#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Send a custom payload (0xDEADBEEF) from AGX Thor to FPGA
# using a Python UDP transmitter (plain UDP -- works on Thor without ConnectX NIC).
#
# UDP datagram payload layout (14 bytes):
#
#   Offset  Size  Field
#   ──────  ────  ──────────────────────────────────────────────
#     0      2    Checksum = 0x0000
#   ── ECB Header (6 bytes) ───────────────────────────────────
#     2      1    Command  = 0x04  (WR_DWORD)
#     3      1    Flags    = 0x01  (bit0: response requested)
#     4      2    Sequence Number = 0x0001  (big-endian)
#     6      2    Padding  = 0x0000
#   ── ECB Data Pair (8 bytes) ────────────────────────────────
#     8      4    Address  = 0x00000000  (glb_scratch, big-endian)
#    12      4    Data     = <payload>   (big-endian)
#   ───────────────────────────────────────────────────────────
#   Total UDP payload: 14 bytes  (NIC pads Ethernet frame to 60 + 4 FCS = 64)
#
# Usage:
#   python3 examples/thor_send_payload.py
#   python3 examples/thor_send_payload.py --hololink 192.168.0.2 --payload 0xCAFEBABE --frame-limit 5 --port 5000
#
# Notes:
#   - Does NOT require the C++ PayloadGeneratorOp build step.
#   - Does NOT require sudo (no raw socket).

import argparse
import logging
import socket
import struct

import numpy as np
import holoscan
import holoscan.resources

import hololink as hololink_module


def build_ecb_packet(payload_value: int, address: int = 0x00000000, seq: int = 1) -> bytes:
    """
    Build a 14-byte ECB packet to be sent as the UDP datagram payload.

    Layout:
      [0:2]   Checksum          = 0x0000
      [2]     Command           = 0x04 (WR_DWORD)
      [3]     Flags             = 0x01 (response requested)
      [4:6]   Sequence Number   (big-endian)
      [6:8]   Padding           = 0x0000
      [8:12]  Address           (big-endian)
      [12:16] Data / payload    (big-endian)
    """
    checksum   = struct.pack(">H", 0x0000)          # 2 bytes
    command    = struct.pack("B",  0x04)             # 1 byte  WR_DWORD
    flags      = struct.pack("B",  0x01)             # 1 byte  response requested
    sequence   = struct.pack(">H", seq & 0xFFFF)    # 2 bytes big-endian
    padding    = struct.pack(">H", 0x0000)           # 2 bytes
    ecb_addr   = struct.pack(">I", address)          # 4 bytes big-endian
    ecb_data   = struct.pack(">I", payload_value)    # 4 bytes big-endian
    return checksum + command + flags + sequence + padding + ecb_addr + ecb_data


class PayloadGeneratorOp(holoscan.core.Operator):
    """
    Builds and emits a 14-byte ECB packet (checksum + ECB header + ECB data pair)
    as a uint8 numpy array every tick.
    """

    def __init__(
        self,
        fragment,
        *args,
        payload_value: int = 0xDEADBEEF,
        address: int = 0x00000000,
        repeat: bool = True,
        **kwargs,
    ):
        self._payload_value = payload_value
        self._address = address
        self._repeat = repeat
        self._sent = False
        self._seq = 1
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: holoscan.core.OperatorSpec):
        spec.output("output")

    def compute(self, op_input, op_output, context):
        if not self._repeat and self._sent:
            return

        packet = build_ecb_packet(self._payload_value, self._address, self._seq)
        data = np.frombuffer(packet, dtype=np.uint8).copy()

        logging.info(
            f"PayloadGeneratorOp: seq={self._seq}  "
            f"addr=0x{self._address:08X}  "
            f"data=0x{self._payload_value:08X}  "
            f"packet({len(packet)}B)={packet.hex(' ')}"
        )

        op_output.emit(data, "output")
        self._sent = True
        self._seq = (self._seq + 1) & 0xFFFF


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
    def __init__(self, hololink_ip, fpga_port, payload_value, address, frame_limit):
        super().__init__()
        self._hololink_ip   = hololink_ip
        self._fpga_port     = fpga_port
        self._payload_value = payload_value
        self._address       = address
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
            address=self._address,
            repeat=True,
        )

        udp_transmitter = PythonUdpTransmitterOp(
            self,
            name="udp_transmitter",
            ip=self._hololink_ip,
            port=self._fpga_port,
        )

        self.add_flow(payload_generator, udp_transmitter, {("output", "input")})


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
        help="32-bit data value to write (default: 0xDEADBEEF)",
    )
    parser.add_argument(
        "--address",
        type=lambda x: int(x, 0),
        default=0x00000000,
        help="32-bit ECB register address to write to (default: 0x00000000, glb_scratch)",
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

    logging.info(
        f"ECB WR_DWORD: addr=0x{args.address:08X}  data=0x{args.payload:08X}  "
        f"→ {args.hololink}:{args.port}"
    )
    logging.info(f"Frame limit: {frame_limit if frame_limit else 'unlimited'}")

    channel_metadata = hololink_module.Enumerator.find_channel(channel_ip=args.hololink)
    logging.info(f"{channel_metadata=}")
    hololink_channel = hololink_module.DataChannel(channel_metadata)

    app = ThorSendPayloadApp(
        hololink_ip=args.hololink,
        fpga_port=args.port,
        payload_value=args.payload,
        address=args.address,
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
