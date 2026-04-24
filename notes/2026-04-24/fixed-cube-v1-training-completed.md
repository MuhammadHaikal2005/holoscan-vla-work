# fixed_cube_v1 — Training Completed & Transferred to Thor

**Date:** 2026-04-24  
**Dataset:** `fixed_cube_v1` — 40 episodes, fixed cube position, cameras and arm base locked down  
**Steps:** 10 000  
**Save interval:** every 1 000 steps  
**Duration:** ~1h31m on RTX 3090 (~1.83–1.99 it/s)  
**Final loss:** 0.0571

---

## Training output (final lines)

```
{'loss': 0.0373, 'grad_norm': 0.760, 'learning_rate': 1.21e-09}
{'loss': 0.0323, 'grad_norm': 0.261, 'learning_rate': 3.31e-10}
{'loss': 0.0340, 'grad_norm': 0.505, 'learning_rate': 2.73e-12}
100%|████████| 10000/10000 [1:30:39<00:00, 1.99it/s]
train_loss: 0.0571  train_samples_per_second: 14.634
04/24/2026 18:26:18 - INFO - Training completed!
```

---

## Available checkpoints on the Thor

Only checkpoints 6 000 – 10 000 were transferred. This is expected behaviour: the
training config has `save_total_limit = 5` (default in `training_config.py`), which
automatically deletes the oldest checkpoint each time a new one is saved. By the time
training finished, checkpoints 1 000 – 5 000 had already been purged on the 3090.

Checkpoints on Thor at:
```
~/hsb-groot-robot/checkpoints/fixed_cube_v1/
    checkpoint-6000/
    checkpoint-7000/
    checkpoint-8000/
    checkpoint-9000/
    checkpoint-10000/
```

Each checkpoint is ~13 GB (two safetensors shards + optimizer state).

---

## Testing order (recommended)

Start at the earliest available checkpoint and work up. Stop when robot performance
stops improving.

| Checkpoint | Test result | Notes |
|---|---|---|
| checkpoint-6000 | | |
| checkpoint-7000 | | |
| checkpoint-8000 | | |
| checkpoint-9000 | | |
| checkpoint-10000 | | |

---

## How to test on the Thor

**Terminal 1 — start the policy server:**
```bash
cd ~/hsb-groot-robot
bash inference/run_server.sh checkpoints/fixed_cube_v1/checkpoint-6000
```

Wait for `Server started on port 5555`, then:

**Terminal 2 — run the robot:**
```bash
conda activate lerobot2
cd ~/hsb-groot-robot
python inference/eval_dual_usb.py
```

To switch checkpoints, stop the server (`Ctrl+C` in Terminal 1) and re-run with a
different checkpoint path.

---

## Training command used

```bash
cd ~/holoscan-vla-work
bash training/finetune_dual_usb.sh fixed_cube_v1 fixed_cube_v1 --max-steps 10000 --save-steps 1000
```

Training was run inside a `tmux` session (`tmux new-session -s training_fixed`) to
prevent the `ble.sh` terminal issue from killing the process mid-run (see
`notes/2026-04-23/groot-training-debug-and-run.md` for full details on that fix).

---

## Key config at time of training

| Setting | Value | Location |
|---|---|---|
| Optimizer | `adamw_bnb_8bit` | `Isaac-GR00T/gr00t/experiment/launch_finetune.py` |
| Gradient checkpointing | `True` | `Isaac-GR00T/gr00t/experiment/launch_finetune.py` |
| Global batch size | 8 | `training/finetune_dual_usb.sh` |
| Dataloader workers | 2 | `training/finetune_dual_usb.sh` |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | `training/finetune_dual_usb.sh` |
| `save_total_limit` | 5 (default) | `Isaac-GR00T/gr00t/configs/training/training_config.py` |

> **Note for future runs:** If you need checkpoints earlier than step 6 000, override
> `save_total_limit` by adding `--save-total-limit <N>` to the training command, or
> increase it permanently in `training_config.py`. For a 10k run with 1k save interval
> you need `--save-total-limit 10` to keep all checkpoints.

---

## Transfer commands used

From the 3090 after training:

```bash
ssh latticeapp@192.168.100.1 "mkdir -p ~/hsb-groot-robot/checkpoints/fixed_cube_v1"

rsync -av --progress \
    ~/holoscan-vla-work/checkpoints/fixed_cube_v1/ \
    latticeapp@192.168.100.1:~/hsb-groot-robot/checkpoints/fixed_cube_v1/
```

Transfer speed: ~265–274 MB/s over direct Ethernet. Total transferred: ~75 GB in ~4.5 minutes.
