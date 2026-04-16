#!/usr/bin/env python3
# Simple UDP listener to test if thor_send_payload.py or linux_audio_player.py
# are successfully transmitting data.
#
# Usage:
#   On the FPGA/target machine (192.168.0.2):
#     python3 test_udp_listener.py --port 4791
#
#   Then run thor_send_payload.py or linux_audio_player.py from Thor.

import argparse
import socket


def main():
    parser = argparse.ArgumentParser(description="Simple UDP listener for testing")
    parser.add_argument("--port", type=int, default=4791, help="UDP port to listen on")
    parser.add_argument("--interface", default="192.69.0.2", help="Interface to bind to")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.interface, args.port))
    
    print(f"Listening on {args.interface}:{args.port}")
    print("Press Ctrl+C to stop\n")

    packet_count = 0
    try:
        while True:
            data, addr = sock.recvfrom(65535)
            packet_count += 1
            print(f"[{packet_count}] Received {len(data)} bytes from {addr[0]}:{addr[1]}")
            print(f"    Hex: {data[:64].hex()}")  # Show first 64 bytes
            if len(data) <= 16:
                print(f"    Full data: {data.hex()}")
    except KeyboardInterrupt:
        print(f"\n\nTotal packets received: {packet_count}")
        sock.close()


if __name__ == "__main__":
    main()
