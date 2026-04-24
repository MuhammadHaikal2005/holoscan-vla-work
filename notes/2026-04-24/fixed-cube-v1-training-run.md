# fixed_cube_v1 — Training Run on 3090

**Date:** 2026-04-24  
**Dataset:** `fixed_cube_v1` — 40 episodes, fixed cube position, cameras and arm base locked down  
**Goal:** Train to 10 000 steps, save every 1000, test each checkpoint to find the optimal step count  
**Previous result:** `dual_cam_blue_only` (30 scattered episodes, 2000 steps) → ~1% success rate  

---

## Step 1 — Ethernet link (do this first)

Plug the Ethernet cable between Thor and the 3090.

**On Thor:**
```bash
sudo ip addr add 192.168.100.1/24 dev enP2p1s0
ping 192.168.100.2 -c 3
```

**On the 3090 (if IP was lost after reboot):**
```bash
# find your interface name first
ip link show | grep -E "LOWER_UP|eno|eth|enp"

# then set the static IP (replace <interface> with your interface name)
sudo ip addr add 192.168.100.2/24 dev <interface>
sudo ip link set <interface> up
```

---

## Step 2 — Transfer dataset from Thor → 3090

Run on the **3090**:

```bash
ssh latticeapp@192.168.100.1 "mkdir -p ~/hsb-groot-robot/datasets/fixed_cube_v1"

rsync -av --progress \
    latticeapp@192.168.100.1:~/hsb-groot-robot/datasets/fixed_cube_v1/ \
    ~/holoscan-vla-work/datasets/fixed_cube_v1/
```

Dataset is ~2–3 GB. Should transfer in under a minute over direct Ethernet.

---

## Step 3 — Run training on the 3090

```bash
cd ~/holoscan-vla-work
tmux new-session -s training_fixed
bash training/finetune_dual_usb.sh fixed_cube_v1 fixed_cube_v1 --max-steps 10000 --save-steps 1000
```

- Detach from tmux: `Ctrl+B  D`
- Reattach later: `tmux attach -t training_fixed`
- Watch progress: `tmux attach -t training_fixed`

**Expected time:** ~50–60 minutes on the 3090  
**Checkpoints saved at:** 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000

Checkpoints will be at:
```
~/holoscan-vla-work/checkpoints/fixed_cube_v1/
    checkpoint-1000/
    checkpoint-2000/
    ...
    checkpoint-10000/
```

---

## Step 4 — Transfer checkpoints back to Thor

Run on the **3090** after training completes:

```bash
ssh latticeapp@192.168.100.1 "mkdir -p ~/hsb-groot-robot/checkpoints/fixed_cube_v1"

rsync -av --progress \
    ~/holoscan-vla-work/checkpoints/fixed_cube_v1/ \
    latticeapp@192.168.100.1:~/hsb-groot-robot/checkpoints/fixed_cube_v1/
```

All 10 checkpoints are ~13 GB each → ~130 GB total. Over direct Ethernet (~1 Gbps) expect ~15–20 minutes.

> **Tip:** If you only want to transfer specific checkpoints first (e.g. just 3000 and 5000):
> ```bash
> for ckpt in checkpoint-3000 checkpoint-5000; do
>     ssh latticeapp@192.168.100.1 "mkdir -p ~/hsb-groot-robot/checkpoints/fixed_cube_v1/$ckpt"
>     rsync -av --progress \
>         ~/holoscan-vla-work/checkpoints/fixed_cube_v1/$ckpt/ \
>         latticeapp@192.168.100.1:~/hsb-groot-robot/checkpoints/fixed_cube_v1/$ckpt/
> done
> ```

---

## Step 5 — Test on Thor

**Terminal 1 — start the policy server:**
```bash
cd ~/hsb-groot-robot
bash inference/run_server.sh checkpoints/fixed_cube_v1/checkpoint-3000
```

Wait for `Server started on port 5555`, then:

**Terminal 2 — run the robot:**
```bash
conda activate lerobot2
cd ~/hsb-groot-robot
python inference/eval_dual_usb.py
```

### Recommended testing order

Start at 3000 steps and work up. Previous dataset converged around 1500 steps — with 40 cleaner episodes it may converge slightly later.

| Checkpoint | Test result | Notes |
|---|---|---|
| checkpoint-3000 | | |
| checkpoint-5000 | | |
| checkpoint-7000 | | |
| checkpoint-10000 | | |

---

## What changed vs the previous dataset

| Factor | `dual_cam_blue_only` | `fixed_cube_v1` |
|---|---|---|
| Episodes | 30 | 40 |
| Cube position | Varied freely | **Fixed on taped marker** |
| Camera position | Not locked | **Locked down** |
| Arm base position | Not locked | **Locked down** |
| Max steps trained | 2000 (then retried 12k/20k) | 10 000 |
| Rerecord bug | Present | **Fixed** |

---

## If training fails to start

**torchcodec check fails:**
```bash
~/Isaac-GR00T/.venv/bin/python3 -c "import torchcodec; print('OK')"
```
If it errors: `sudo apt install ffmpeg`

**CUDA not visible:**
```bash
~/Isaac-GR00T/.venv/bin/python3 -c "import torch; print(torch.cuda.is_available())"
```

**Out of memory (unlikely on 3090 with 8-bit Adam):**  
Reduce batch size in `training/finetune_dual_usb.sh`: `--global-batch-size 4`
