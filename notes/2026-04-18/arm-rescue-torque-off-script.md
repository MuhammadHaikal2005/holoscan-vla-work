# Arm Rescue — torque_off.py
**Date:** 2026-04-18  
**Session goal:** Handle the situation where the recording script crashes mid-startup, leaving one or both arms locked with torque enabled and no way to move them.

---

## Problem

While resuming a recording session the script crashed with:

```
ConnectionError: Failed to write 'Torque_Enable' on id_=1 with '0' after 1 tries.
[TxRxResult] Incorrect status packet!
```

The error occurred inside `_create_and_connect_bus` when trying to disable torque on the **leader arm**. The follower arm had already connected successfully with torque **ON**, so after the crash:

- **Follower** — torque ON, locked rigid at its last position
- **Leader** — state unknown, possibly also stuck

There was no way to move either arm by hand without first sending torque-off commands over serial.

### Root cause

The leader arm returned `Incorrect status packet` when the script tried to write to it. This is typically caused by:
- Flaky USB cable or loose connector on the leader arm
- Wrong USB port (data-only vs. data+power)
- Serial bus collision if another process had the port open

---

## Fix — `recording/torque_off.py`

A standalone rescue script was created that sends raw Feetech torque-off packets directly over serial, **without any LeRobot dependency**. It works even if the lerobot conda environment is broken or unavailable.

```python
# Sends torque-off (register 0x28 = 0) to motor IDs 1–6 on both ports
PORTS     = ["/dev/ttyACM0", "/dev/ttyACM1"]
MOTOR_IDS = range(1, 7)
```

Each motor receives the packet `[0xFF, 0xFF, id, 0x04, 0x03, 0x28, 0x00, checksum]`.

### Usage

```bash
python recording/torque_off.py
```

No conda activation needed — only `pyserial` is required (installed system-wide).  
If a port is not connected the script skips it with a warning and continues.

---

## How to avoid the crash in future

The `_create_and_connect_bus` function in `imx274_lerobot_record.py` already has retry logic for reads, but the initial torque-write on connect has no retry. If the leader arm returns a bad packet at startup, the whole script aborts and leaves the follower locked.

**Recommended hardware checks before each session:**
1. Reseat the USB cables on both arms — leader failures are almost always physical
2. Prefer powered USB hubs or the Jetson's USB-A ports (not USB-C adapters)
3. Check `ls /dev/ttyACM*` before starting — both arms should appear

---

## Summary of files changed

| File | Change |
|---|---|
| `recording/torque_off.py` | New rescue script — raw serial torque-off for both arms |
