# Transferring Training to the 3090

**Date:** 2026-04-22  
**Context:** Dual-cam training (`dual_cam_blue_only`) was running on the Jetson AGX Thor but kept hitting over-current throttling (~100W total draw against a 120W PSU). The Thor is better suited for inference; the 3090 is the right machine for training. This note covers what to transfer, how to do it, and how to resume on the 3090.

---

## Why move to the 3090?

| | Jetson AGX Thor | RTX 3090 |
|---|---|---|
| GPU TDP | ~40W (training) | ~350W |
| Training speed (dual-cam) | ~3 s/step | ~0.3–0.5 s/step (estimated) |
| Ecosystem | aarch64 — custom wheel builds, FFmpeg mismatches | x86_64 — standard PyPI wheels, no custom builds |
| Over-current risk | Yes (PSU ceiling ~105W) | No |
| Best use | Inference / robot control | Training |

Fine-tuning on different hardware than inference is fine — GR00T weights are bfloat16 and hardware-agnostic. Train on the 3090, copy `checkpoint-2000/` back to Thor for inference.

---

## What needs to be transferred

| Item | Size | Notes |
|---|---|---|
| `datasets/dual_cam_blue_only/` | 1.4 GB | Converted v2.1 format |
| `models/base/GR00T-N1.6-3B/` | 9.2 GB | Base model weights |
| `checkpoints/dual_cam_blue_only/checkpoint-500/` | ~22 GB | Resume point — don't skip |
| `Isaac-GR00T/` (no `.venv`, no `wheels/`) | 2.8 GB | Codebase only |
| `training/` scripts | 60 MB | finetune scripts, configs, converter |
| **Total** | **~36 GB** | |

**Do NOT copy:**
- `Isaac-GR00T/.venv/` — aarch64 binaries, useless on x86_64. Rebuild with `uv sync` on the 3090.
- `Isaac-GR00T/wheels/` — the custom torchcodec aarch64 wheel. x86_64 uses the standard PyPI wheel.
- `checkpoints/dual_cam_blue_only/checkpoint-1000/` and later — only `checkpoint-500` is needed to resume.

---

## USB transfer — not enough space

The available USB (`/media/latticeapp/BACKUP`, 29 GB total) only had 15 GB free — not enough for 36 GB. The base model alone is 9.2 GB and the checkpoint is 22 GB.

---

## Recommended: direct Ethernet cable transfer

The Thor and 3090 are not on the same network. The fastest solution with no USB size constraints: plug an Ethernet cable directly between the two machines and assign static IPs. No router needed.

### Step 1 — Assign static IPs

**On the Thor:**
```bash
sudo ip addr add 192.168.100.1/24 dev eth0
sudo ip link set eth0 up
```

**On the 3090:**
```bash
sudo ip addr add 192.168.100.2/24 dev eth0
sudo ip link set eth0 up
```

Verify the link is up on both sides:
```bash
ping 192.168.100.2   # run from Thor
```

### Step 2 — Ensure SSH is running on the 3090

```bash
# On the 3090
sudo systemctl start ssh
sudo systemctl enable ssh   # optional: start on boot
```

### Step 3 — Transfer from the Thor

```bash
# Code (no venv, no wheels — rebuild on 3090 with uv sync)
rsync -av --progress --exclude='.venv' --exclude='wheels' \
    ~/Isaac-GR00T/ latticeapp@192.168.100.2:~/Isaac-GR00T/

# Dataset
rsync -av --progress \
    ~/hsb-groot-robot/datasets/dual_cam_blue_only/ \
    latticeapp@192.168.100.2:~/hsb-groot-robot/datasets/dual_cam_blue_only/

# Base model
rsync -av --progress \
    ~/hsb-groot-robot/models/base/GR00T-N1.6-3B/ \
    latticeapp@192.168.100.2:~/hsb-groot-robot/models/base/GR00T-N1.6-3B/

# Checkpoint-500 only
rsync -av --progress \
    ~/hsb-groot-robot/checkpoints/dual_cam_blue_only/checkpoint-500/ \
    latticeapp@192.168.100.2:~/hsb-groot-robot/checkpoints/dual_cam_blue_only/checkpoint-500/

# Training scripts and configs
rsync -av --progress \
    ~/hsb-groot-robot/training/ \
    latticeapp@192.168.100.2:~/hsb-groot-robot/training/
```

At gigabit speed (~100 MB/s), 36 GB takes about **6 minutes**.

---

## Setting up on the 3090

### Step 1 — Rebuild the venv

```bash
cd ~/Isaac-GR00T
uv sync
```

On x86_64, `uv sync` pulls standard PyPI wheels. No custom torchcodec build needed — the standard wheel ships the right FFmpeg version for x86_64 Linux.

### Step 2 — Fix the torchcodec check in finetune scripts

`finetune_dual_usb.sh` contains a self-healing check for the aarch64 torchcodec build:
```bash
TC_CORE6="...libtorchcodec_core6.so"
if [[ ! -f "$TC_CORE6" ]]; then
    uv pip install ... "$TC_WHEEL"   # installs aarch64 wheel — wrong on x86_64
fi
```

On x86_64 the venv will have `libtorchcodec_core6.so` or `libtorchcodec_core7.so` depending on the system FFmpeg version — either way the PyPI wheel is correct and the check should not run. The safest fix: replace the check with a Python import test:

```bash
# Replace the TC_CORE6 check block with:
if ! "$VENV_PYTHON" -c "import torchcodec" 2>/dev/null; then
    echo "torchcodec: not loadable — check FFmpeg installation"
    exit 1
fi
```

Or just delete the check block entirely on the 3090 — uv will manage the correct wheel.

### Step 3 — Resume training

```bash
cd ~/hsb-groot-robot
bash training/finetune_dual_usb.sh dual_cam_blue_only dual_cam_blue_only
```

No `--resume` flag needed. The trainer detects `checkpoints/dual_cam_blue_only/checkpoint-500/` automatically and resumes from step 500.

---

## After training completes on the 3090

Copy the final checkpoint back to Thor for inference:

```bash
# From the 3090 back to Thor
rsync -av --progress \
    ~/hsb-groot-robot/checkpoints/dual_cam_blue_only/checkpoint-2000/ \
    latticeapp@192.168.100.1:~/hsb-groot-robot/checkpoints/dual_cam_blue_only/checkpoint-2000/
```

Then on Thor, point the inference server at:
```
/home/latticeapp/hsb-groot-robot/checkpoints/dual_cam_blue_only/checkpoint-2000
```

---

## Checklist for tomorrow

- [ ] Plug Ethernet cable between Thor and 3090
- [ ] Assign static IPs on both machines
- [ ] Start SSH on the 3090 (`sudo systemctl start ssh`)
- [ ] Ping to verify the link works
- [ ] Run the 5 rsync commands from Thor
- [ ] On 3090: `cd ~/Isaac-GR00T && uv sync`
- [ ] Fix or remove the aarch64 torchcodec check in `finetune_dual_usb.sh`
- [ ] Run `bash training/finetune_dual_usb.sh dual_cam_blue_only dual_cam_blue_only`
- [ ] Confirm training resumes from step 500 (check first log line for "global step: 500")
