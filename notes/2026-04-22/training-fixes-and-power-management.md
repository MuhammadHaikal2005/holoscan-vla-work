# Training Fixes and Power Management

**Date:** 2026-04-22  
**Context:** Continuing from the dual-USB-camera training debug log. This covers the three version mismatches that blocked `control_blue_only` training from starting, torchcodec being repeatedly reset by uv, dual-cam training getting stuck during shard caching, and the over-current throttling issue.

---

## 1. Diagnosing why `finetune.sh control_blue_only` failed

The training was aborting with no Python traceback — just `Aborted` and exit code 1. The question was whether this was the version mismatch suspected from an earlier session.

There were actually **three separate mismatches**, all needed to be fixed before training could start.

---

## 2. Mismatch #1 — LeRobot v3.0 vs v2.1 dataset format

### Problem
GR00T's `LeRobotEpisodeLoader` expects the older **v2.1** dataset format:
- `meta/episodes.jsonl`
- `meta/tasks.jsonl`
- one parquet per episode at `data/chunk-NNN/episode_NNNNNN.parquet`
- one video per episode at `videos/chunk-NNN/<key>/episode_NNNNNN.mp4`

The recorder (LeRobot library) writes **v3.0** format:
- episode metadata stored in `meta/episodes/chunk-000/file-000.parquet`
- multiple episodes packed into `data/chunk-NNN/file-NNN.parquet`
- videos at `videos/<key>/chunk-NNN/file-NNN.mp4`

The first error hit was `FileNotFoundError: meta/episodes.jsonl`. A quick workaround had generated `episodes.jsonl` and `tasks.jsonl` but that only fixed the metadata — the data and video paths were still wrong.

### Fix — `training/convert_v3_to_v2.py`
A full conversion script that:
1. Reads each episode's rows from the multi-episode v3 parquets and writes one `.parquet` per episode
2. Copies each video file from the v3 path structure to the v2.1 path structure
3. Rewrites `info.json` with v2.1 path templates (`data_path`, `video_path`)
4. Generates `episodes.jsonl` and `tasks.jsonl`
5. Backs up the original dataset as `<name>_v30` before swapping

Run it (requires `lerobot2` conda env for `pyarrow`):
```bash
conda activate lerobot2
python training/convert_v3_to_v2.py datasets/control_blue_only
```

The converted dataset sits at the original path; the v3 backup is at `datasets/control_blue_only_v30`.

---

## 3. Mismatch #2 — torchcodec built against FFmpeg 7, system has FFmpeg 6

### Problem
Training started after the format fix but immediately hit:
```
Video backend 'torchcodec' is not available, falling back to 'pyav'.
...
NotImplementedError in get_frames_by_indices
```

GR00T's video loader uses `torchcodec` as its primary backend. The `torchcodec` wheel installed in Isaac-GR00T's `.venv` was compiled against **FFmpeg 7** (ships `libtorchcodec_core7.so` which links to `libavfilter.so.10`). The system (Ubuntu 24.04 / JetPack 6) only has **FFmpeg 6** (`libavfilter.so.9`).

GR00T falls back to `pyav` when torchcodec fails, but the `pyav` branch in `get_frames_by_indices` hits `raise NotImplementedError`.

Symlinking `libavfilter.so.10 → libavfilter.so.9` was considered but ruled out: `libtorchcodec_core7.so` uses **versioned ELF symbols** (`@LIBAVCODEC_61`, `@LIBAVUTIL_59`) and calls functions added in FFmpeg 7 (e.g. `avcodec_get_supported_config`) that literally do not exist in FFmpeg 6. A symlink would load then segfault.

### Root cause of why it kept reverting
`pyproject.toml` in Isaac-GR00T pins torchcodec to NVIDIA's Jetson-specific PyPI index:
```
torchcodec = [{ index = "jetson-sbsa-cu130" }]
```
That index hosts a pre-built aarch64 wheel that was compiled against FFmpeg 7. Every time `uv sync` or `uv run` ran, it restored the FFmpeg 7 wheel from that index.

### Fix — build from source + cache the wheel
Build torchcodec from source against the system's FFmpeg 6. The cmake build script detects FFmpeg version via `pkg-config` (libavcodec 60.x → FFmpeg 6) and produces `libtorchcodec_core6.so`.

```bash
cd ~/Isaac-GR00T
SITE_PKGS="$(pwd)/.venv/lib/python3.12/site-packages"
NVIDIA_LIB_DIRS="$(find "${SITE_PKGS}/nvidia" -name "lib" -type d | tr '\n' ':')"
export LD_LIBRARY_PATH="${SITE_PKGS}/torch/lib:${NVIDIA_LIB_DIRS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME=/usr/local/cuda-13.0 CUDA_PATH=/usr/local/cuda-13.0
export CPATH="${CUDA_HOME}/include" CPLUS_INCLUDE_PATH="${CUDA_HOME}/include"

git clone --depth 1 --branch v0.10.0 https://github.com/pytorch/torchcodec.git /tmp/torchcodec
cd /tmp/torchcodec

# Build and save the wheel (takes ~1m40s)
I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1 \
  uv build --wheel --python ~/Isaac-GR00T/.venv/bin/python3 \
  --no-build-isolation -o ~/Isaac-GR00T/wheels/
```

The wheel is saved permanently at `Isaac-GR00T/wheels/torchcodec-0.10.0a0-cp312-cp312-linux_aarch64.whl` and contains `libtorchcodec_core6.so`.

---

## 4. Mismatch #3 — `uv run` kept restoring the PyPI wheel

### Problem
Even after building from source, using `uv run python` to launch training caused uv to re-resolve dependencies from the lockfile and reinstall the Jetson PyPI wheel (FFmpeg 7), wiping our FFmpeg 6 build.

### Fix — two-part solution

**Part A:** Both `finetune.sh` and `finetune_dual_usb.sh` now use the venv's python directly instead of `uv run`:
```bash
VENV_PYTHON="$GROOT_DIR/.venv/bin/python3"
CUDA_VISIBLE_DEVICES=0 "$VENV_PYTHON" gr00t/experiment/launch_finetune.py ...
```

**Part B:** A self-healing check at the top of each finetune script detects if uv has reset torchcodec (checks for `libtorchcodec_core6.so`) and reinstalls from the cached wheel in ~3 seconds if so:
```bash
TC_WHEEL="$GROOT_DIR/wheels/torchcodec-0.10.0a0-cp312-cp312-linux_aarch64.whl"
TC_CORE6="$GROOT_DIR/.venv/lib/python3.12/site-packages/torchcodec/libtorchcodec_core6.so"
if [[ ! -f "$TC_CORE6" ]]; then
    echo "torchcodec: FFmpeg-6 build missing — reinstalling from cached wheel..."
    uv pip install --python "$VENV_PYTHON" --no-deps "$TC_WHEEL"
fi
```

This is permanent and survives reboots and `uv sync` calls without any manual intervention.

---

## 5. `control_blue_only` training — first complete run

With all three fixes in place, training started successfully:
- ~1.1 s/step for 2000 steps ≈ 37 minutes
- Checkpoints saved every 500 steps to `checkpoints/control_blue_only/`
- **Training completed** — `checkpoint-2000/` contains the final fine-tuned model

### Reboot mid-run
Partway through the first attempt (~step 252), nvpmodel sent an over-current notification. The user wanted to change the power mode, which requires a reboot on Thor.

Since no checkpoint had been saved yet (first save is at step 500), the run was lost. After reboot:
```bash
cd ~/hsb-groot-robot && bash training/finetune.sh control_blue_only control_blue_only
```
No `--resume` flag needed — the trainer always calls `resume_from_checkpoint=True` internally and auto-detects the latest checkpoint in the output directory. Since there were none, it started fresh.

The second attempt ran to completion without interruption.

---

## 6. `--resume` flag does not exist

When trying to pass `--resume` as a CLI argument:
```
╭─ Unrecognized options ───────────────────────────╮
│ Unrecognized options: --resume                   │
╰──────────────────────────────────────────────────╯
```

`launch_finetune.py` does not accept `--resume`. The HuggingFace Trainer always resumes from the latest `checkpoint-XXXX/` folder in the output directory automatically. To resume, just re-run the same command — no extra flags.

---

## 7. Dual-cam training — opencv monkey-patch removed

The earlier session (see `dual-usb-camera-training-debug-log.md`) used a `launch_finetune_ffmpeg.py` wrapper that monkey-patched GR00T's video loader to use opencv. With the torchcodec fix now in place, that wrapper was no longer needed.

`finetune_dual_usb.sh` was updated to call `launch_finetune.py` directly (same as `finetune.sh`) instead of going through the wrapper. The opencv warnings disappeared entirely.

---

## 8. Shard caching phase — "looks stuck, is actually working"

After launching dual-cam training, the terminal showed:
```
Rank 0, Worker 0: Caching shard...
Rank 0, Worker 1: Caching shard...
Rank 0, Worker 2: Caching shard...
Rank 0, Worker 3: Caching shard...
```
…and nothing else for several minutes. No progress bar.

This is **normal**. GR00T pre-caches all dataset shards into memory before training starts. Workers were confirmed to be actively running at 183–189% CPU each. The progress bar only appears after all workers finish their assigned shards.

With the dual-cam dataset (10,815 frames × 2 cameras) and 4 workers, caching took roughly 5–8 minutes before the first `it/s` appeared. After that, training ran at ~2.98 s/step.

The shard wait messages like `Wait for shard 9 in dataset 0 in 94.29 seconds` are also normal — workers wait for their next shard to be pre-cached before proceeding.

---

## 9. Over-current throttling — cause and mitigation

### What triggers it
The over-current notification comes from nvpmodel detecting that total system draw is hitting the PSU's continuous delivery ceiling. Measured power breakdown during training:

| Component | Draw |
|---|---|
| GPU (model forward/backward) | ~38W |
| CPU (workers + OS) | ~20W |
| Memory / IO / SoC | ~30W |
| Peripherals | ~15W |
| **Total** | **~100–103W** |

The Thor dev kit PSU is rated at 120W, but PSUs typically trigger OCP at 90–95% of their nameplate (≈108–114W) to protect themselves. Training peaks above that threshold on spikes.

### Is it dangerous?
No. The OCP is a protection circuit that does exactly what it should — throttle the system before anything can be damaged. The hardware is safe.

### Software mitigations (no reboot needed)
1. **Reduce workers**: `--dataloader-num-workers 1` drops CPU draw from ~20W to ~6W
2. **Reduce batch size**: `--global-batch-size 8` instead of 16 cuts peak GPU memory bandwidth

```bash
nohup bash training/finetune_dual_usb.sh dual_cam_blue_only dual_cam_blue_only \
    --dataloader-num-workers 1 \
    --global-batch-size 8 \
    > /tmp/train.log 2>&1 &
```

Expected: ~5–6 s/step instead of ~3 s/step (≈3.5 hours for 2000 steps), but uninterrupted.

### Hardware fix (permanent)
A 150–180W 19V barrel-jack PSU (same connector as the stock one) gives 50W of headroom. Training can run at full settings without ever throttling.

### Power mode change (requires reboot)
```bash
sudo nvpmodel --list-modes   # see available modes and their watt caps
sudo nvpmodel -m 2           # example: switch to a 60W-capped mode
```
On Jetson AGX Thor, nvpmodel changes require a reboot to take effect.

---

## Summary of files changed this session

| File | Change |
|---|---|
| `training/convert_v3_to_v2.py` | **New.** Full LeRobot v3.0 → v2.1 converter (parquets, videos, info.json, episodes.jsonl, tasks.jsonl). |
| `training/finetune.sh` | Use `.venv/bin/python3` directly (not `uv run`); add self-healing torchcodec check. |
| `training/finetune_dual_usb.sh` | Same torchcodec check added; switched from `launch_finetune_ffmpeg.py` wrapper back to `launch_finetune.py`. |
| `Isaac-GR00T/wheels/torchcodec-0.10.0a0-cp312-cp312-linux_aarch64.whl` | **New.** torchcodec built from source against system FFmpeg 6. Cached so reinstallation takes ~3 s. |

## Datasets trained

| Dataset | Status | Checkpoint |
|---|---|---|
| `control_blue_only` | ✅ Complete | `checkpoints/control_blue_only/checkpoint-2000` |
| `dual_cam_blue_only` | 🔄 In progress | `checkpoints/dual_cam_blue_only/` |

## End-to-end training command (current)

```bash
# Convert dataset (only needed once per recording session)
conda activate lerobot2
python training/convert_v3_to_v2.py datasets/<dataset_folder>
conda deactivate

# Train (self-heals torchcodec on every run)
cd ~/hsb-groot-robot
nohup bash training/finetune_dual_usb.sh <dataset_folder> <checkpoint_folder> \
    > /tmp/train.log 2>&1 &

# Monitor
tail -f /tmp/train.log | tr '\r' '\n' | grep --line-buffered "it/s\|loss\|checkpoint"
```
