# Setting Up the 3090 for GR00T Training

**Date:** 2026-04-22  
**Context:** Transferring dual-cam GR00T training from the Jetson AGX Thor to the 3090 via direct Ethernet cable. The Thor and 3090 are not on the same network, so a static-IP direct link is used.

---

## Step 1 — Find the Ethernet interface on the 3090

```bash
ip link show
```

Look for an interface that starts with `eth`, `enp`, `eno`, or `ens`. It will show `state DOWN` if the cable is plugged in but not configured. Note the name — e.g. `enp5s0`.

---

## Step 2 — Bring up the direct link on the 3090

Replace `enp5s0` with your actual interface name:

```bash
sudo ip addr add 192.168.100.2/24 dev enp5s0
sudo ip link set enp5s0 up
```

Verify it's up:
```bash
ip addr show enp5s0
```

You should see `192.168.100.2/24` listed and `state UP`.

---

## Step 3 — Start SSH on the 3090

```bash
sudo systemctl start ssh

# Optional: make it start automatically on boot
sudo systemctl enable ssh
```

---

## Step 4 — Set the IP on the Thor

On the Thor, first check which interface lit up when the cable was connected:
```bash
ip link show | grep -E "mgbe|UP"
```

The one that changed from `NO-CARRIER` to `UP` is the right one (e.g. `mgbe0_0`). Then:

```bash
sudo ip addr add 192.168.100.1/24 dev mgbe0_0
sudo ip link set mgbe0_0 up
```

---

## Step 5 — Verify the link works

From the Thor:
```bash
ping 192.168.100.2
```

You should see replies. If not, check that both sides have their IPs set and the cable is firmly seated.

---

## Step 6 — Transfer files from the Thor

Run these from the Thor. Total transfer is ~36 GB — takes about 6 minutes on gigabit.

```bash
# Isaac-GR00T codebase (no venv — will be rebuilt on 3090)
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

# Checkpoint-500 (resume point)
rsync -av --progress \
    ~/hsb-groot-robot/checkpoints/dual_cam_blue_only/checkpoint-500/ \
    latticeapp@192.168.100.2:~/hsb-groot-robot/checkpoints/dual_cam_blue_only/checkpoint-500/

# Training scripts and configs
rsync -av --progress \
    ~/hsb-groot-robot/training/ \
    latticeapp@192.168.100.2:~/hsb-groot-robot/training/
```

> **Note:** Replace `latticeapp` with your username on the 3090 if it is different.

---

## Step 7 — Set up the Python environment on the 3090

```bash
cd ~/Isaac-GR00T

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

# Build the venv — on x86_64 this pulls standard PyPI wheels, no custom builds needed
uv sync
```

---

## Step 8 — Fix the aarch64 torchcodec check in finetune_dual_usb.sh

The finetune script contains a self-healing check that was written for the Thor (aarch64 / FFmpeg 6). On x86_64, the standard torchcodec wheel from PyPI works fine and the check will try to install an incompatible ARM wheel.

Open `~/hsb-groot-robot/training/finetune_dual_usb.sh` and find this block:

```bash
TC_WHEEL="$GROOT_DIR/wheels/torchcodec-0.10.0a0-cp312-cp312-linux_aarch64.whl"
TC_CORE6="$GROOT_DIR/.venv/lib/python3.12/site-packages/torchcodec/libtorchcodec_core6.so"
if [[ ! -f "$TC_CORE6" ]]; then
    echo "torchcodec: FFmpeg-6 build missing — reinstalling from cached wheel..."
    uv pip install --python "$VENV_PYTHON" --no-deps "$TC_WHEEL"
    echo "torchcodec: reinstalled OK"
fi
```

Replace it with a simple import test that works on any architecture:

```bash
if ! "$VENV_PYTHON" -c "import torchcodec" 2>/dev/null; then
    echo "ERROR: torchcodec failed to load. Check FFmpeg installation."
    echo "  On x86_64: uv sync should have installed the correct wheel."
    echo "  Try: sudo apt install ffmpeg"
    exit 1
fi
```

Do the same for `finetune.sh` if you plan to use that script too.

---

## Step 9 — Resume training from checkpoint-500

```bash
cd ~/hsb-groot-robot
bash training/finetune_dual_usb.sh dual_cam_blue_only dual_cam_blue_only
```

No `--resume` flag needed. The HuggingFace Trainer automatically detects
`checkpoints/dual_cam_blue_only/checkpoint-500/` and resumes from step 500.

Confirm in the first few lines of output:
```
Current global step: 500
```

At ~0.3–0.5 s/step on the 3090, the remaining 1500 steps will finish in **8–12 minutes**.

---

## Step 10 — Copy the final checkpoint back to Thor for inference

After training completes, transfer `checkpoint-2000` back to the Thor:

```bash
# From the 3090
rsync -av --progress \
    ~/hsb-groot-robot/checkpoints/dual_cam_blue_only/checkpoint-2000/ \
    latticeapp@192.168.100.1:~/hsb-groot-robot/checkpoints/dual_cam_blue_only/checkpoint-2000/
```

Then on the Thor, point the inference server at:
```
/home/latticeapp/hsb-groot-robot/checkpoints/dual_cam_blue_only/checkpoint-2000
```

---

## Checklist

- [ ] Find Ethernet interface name on 3090 (`ip link show`)
- [ ] Set static IP on 3090 (`192.168.100.2/24`)
- [ ] Start SSH on 3090 (`sudo systemctl start ssh`)
- [ ] Find which `mgbeX_0` interface lit up on Thor
- [ ] Set static IP on Thor (`192.168.100.1/24`)
- [ ] Ping 192.168.100.2 from Thor — confirm replies
- [ ] Run 5 rsync commands from Thor (~6 min)
- [ ] On 3090: `cd ~/Isaac-GR00T && uv sync`
- [ ] Fix torchcodec check in `finetune_dual_usb.sh`
- [ ] Run `bash training/finetune_dual_usb.sh dual_cam_blue_only dual_cam_blue_only`
- [ ] Confirm output says `Current global step: 500`
- [ ] After training: rsync `checkpoint-2000` back to Thor
