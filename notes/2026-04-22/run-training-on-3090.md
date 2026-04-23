# Running GR00T Training on the 3090

**Date:** 2026-04-22  
**Context:** Files have been transferred from the Jetson AGX Thor to the 3090 via direct Ethernet. This note covers everything needed to get the environment set up and training running from scratch.

## What was transferred and where it landed

| Item | Path on 3090 |
|---|---|
| Project repo (cloned from GitHub) | `~/holoscan-vla-work/` |
| Isaac-GR00T codebase (no venv) | `~/Isaac-GR00T/` |
| Dataset | `~/holoscan-vla-work/datasets/dual_cam_blue_only/` |
| Base model | `~/holoscan-vla-work/models/base/GR00T-N1.6-3B/` |
| Training scripts | `~/holoscan-vla-work/training/` |

---

## Step 1 — Create a symlink so the training scripts find the right paths

The training scripts hardcode `$HOME/hsb-groot-robot` as the project directory (that was the folder name on the Thor). On the 3090 the repo cloned as `holoscan-vla-work`. The easiest fix is a symlink — no script editing needed:

```bash
ln -s ~/holoscan-vla-work ~/hsb-groot-robot
```

Verify:
```bash
ls ~/hsb-groot-robot/training/
```

You should see `finetune_dual_usb.sh`, `convert_v3_to_v2.py`, etc.

---

## Step 2 — Build the Python environment

```bash
cd ~/Isaac-GR00T

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env   # or open a new terminal

# Build the venv — pulls standard x86_64 PyPI wheels, no custom builds needed
uv sync
```

This will take a few minutes the first time. On x86_64 all wheels (including torchcodec) install cleanly from PyPI — none of the FFmpeg 6/7 mismatch issues from the Thor.

---

## Step 3 — Fix the aarch64 torchcodec check in finetune_dual_usb.sh

The training script has a self-healing block written for the Thor (aarch64 / FFmpeg 6 custom wheel). On x86_64 this block will try to install an ARM64 wheel and fail. Replace it with a simple import test.

Open the file:
```bash
nano ~/hsb-groot-robot/training/finetune_dual_usb.sh
```

Find this block (around line 75):
```bash
TC_WHEEL="$GROOT_DIR/wheels/torchcodec-0.10.0a0-cp312-cp312-linux_aarch64.whl"
TC_CORE6="$GROOT_DIR/.venv/lib/python3.12/site-packages/torchcodec/libtorchcodec_core6.so"
if [[ ! -f "$TC_CORE6" ]]; then
    echo "torchcodec: FFmpeg-6 build missing — reinstalling from cached wheel..."
    uv pip install --python "$VENV_PYTHON" --no-deps "$TC_WHEEL"
    echo "torchcodec: reinstalled OK"
fi
```

Replace it with:
```bash
if ! "$VENV_PYTHON" -c "import torchcodec" 2>/dev/null; then
    echo "ERROR: torchcodec failed to load. Try: sudo apt install ffmpeg"
    exit 1
fi
```

Save and exit (`Ctrl+O`, `Enter`, `Ctrl+X` in nano).

---

## Step 4 — Verify the dataset is in v2.1 format

The dataset was already converted on the Thor. Confirm:
```bash
ls ~/hsb-groot-robot/datasets/dual_cam_blue_only/meta/
```

You should see `episodes.jsonl`, `tasks.jsonl`, `info.json`, and `modality.json`. If you only see parquet files, the conversion didn't transfer — see the note on convert_v3_to_v2.py below.

---

## Step 5 — Run training

```bash
cd ~/hsb-groot-robot
bash training/finetune_dual_usb.sh dual_cam_blue_only dual_cam_blue_only
```

Training starts from step 0 (no valid checkpoint was saved on the Thor before it was stopped). At ~0.3–0.5 s/step on a 3090, **2000 steps takes about 10–17 minutes**.

Watch for this in the output confirming it started correctly:
```
Current global step: 0
Creating custom train dataloader
  0%|          | 0/2000 ...
```

And this confirming the first step worked:
```
{'loss': ..., 'grad_norm': ..., 'learning_rate': ...}
  0%|          | 1/2000 [00:00<..., X.XXs/it]
```

---

## Step 6 — Monitor training in a second terminal

```bash
# In a separate terminal on the 3090
tail -f /tmp/train.log | tr '\r' '\n' | grep --line-buffered "it/s\|loss\|checkpoint"
```

(Only works if you launched with `nohup ... > /tmp/train.log 2>&1 &`. Otherwise just watch the terminal directly.)

Checkpoints save every 500 steps to:
```
~/hsb-groot-robot/checkpoints/dual_cam_blue_only/checkpoint-500/
~/hsb-groot-robot/checkpoints/dual_cam_blue_only/checkpoint-1000/
...
```

---

## Step 7 — After training completes

The final model is at:
```
~/hsb-groot-robot/checkpoints/dual_cam_blue_only/checkpoint-2000/
```

To transfer it back to the Thor for inference (re-establish the direct Ethernet link first):
```bash
# On the 3090 — send checkpoint back to Thor
rsync -av --progress \
    ~/hsb-groot-robot/checkpoints/dual_cam_blue_only/checkpoint-2000/ \
    latticeapp@192.168.100.1:~/hsb-groot-robot/checkpoints/dual_cam_blue_only/checkpoint-2000/
```

On the Thor, point the inference server at:
```
/home/latticeapp/hsb-groot-robot/checkpoints/dual_cam_blue_only/checkpoint-2000
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'gr00t'`
You're not running from the right directory. Always `cd ~/Isaac-GR00T` before running, or the script does it for you via `cd "$GROOT_DIR"`.

### `FileNotFoundError: meta/episodes.jsonl`
The dataset is still in v3.0 format. Convert it:
```bash
conda activate lerobot2   # or whichever env has pandas + pyarrow
python ~/hsb-groot-robot/training/convert_v3_to_v2.py \
    ~/hsb-groot-robot/datasets/dual_cam_blue_only
```

### `ERROR: Missing modality.json`
The file should already exist (it was in the transfer). If not, create it at `~/hsb-groot-robot/datasets/dual_cam_blue_only/meta/modality.json`:
```json
{
  "video": {
    "cam0": {"original_key": "observation.images.cam0"},
    "cam1": {"original_key": "observation.images.cam1"}
  },
  "state": {
    "single_arm": {"start": 0, "end": 5},
    "gripper":    {"start": 5, "end": 6}
  },
  "action": {
    "single_arm": {"start": 0, "end": 5, "absolute": false},
    "gripper":    {"start": 5, "end": 6, "absolute": true}
  },
  "language": {"task_description": "annotation.human.task_description"}
}
```

### torchcodec warning spam
If you see `Video backend 'torchcodec' is not available` warnings, FFmpeg may not be installed:
```bash
sudo apt install ffmpeg
```
Then rerun `uv sync` in `~/Isaac-GR00T`.

---

## Quick checklist

- [ ] `ln -s ~/holoscan-vla-work ~/hsb-groot-robot`
- [ ] `cd ~/Isaac-GR00T && uv sync`
- [ ] Fix aarch64 torchcodec block in `finetune_dual_usb.sh`
- [ ] Verify `meta/episodes.jsonl` exists in the dataset
- [ ] `bash training/finetune_dual_usb.sh dual_cam_blue_only dual_cam_blue_only`
- [ ] Confirm `0%| 1/2000` appears and loss is printing
- [ ] After ~15 min: `checkpoint-2000` is saved
- [ ] rsync `checkpoint-2000` back to Thor
