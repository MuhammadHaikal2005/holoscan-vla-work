"""Quick rescue script — releases all servos on both arms using raw serial.

No LeRobot dependency needed. Run any time an arm gets stuck with torque on.

Usage:
    python recording/torque_off.py
"""
import serial
import time

PORTS      = ["/dev/ttyACM0", "/dev/ttyACM1"]
BAUD_RATE  = 1000000
MOTOR_IDS  = range(1, 7)   # IDs 1–6 (all joints)


def release_servo(ser, servo_id):
    packet   = [servo_id, 0x04, 0x03, 0x28, 0x00]
    checksum = (~sum(packet)) & 0xFF
    ser.write(bytearray([0xFF, 0xFF] + packet + [checksum]))
    time.sleep(0.02)


for port in PORTS:
    print(f"\n── {port} ──")
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=0.1)
    except Exception as e:
        print(f"  Could not open port: {e}")
        continue

    for motor_id in MOTOR_IDS:
        release_servo(ser, motor_id)
        print(f"  torque OFF: id={motor_id}")

    ser.close()
    print(f"  done.")

print("\nAll arms should now be compliant.")
