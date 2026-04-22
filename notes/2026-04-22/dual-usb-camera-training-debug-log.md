# Dual-USB-Camera GR00T Training — Debug Log

Date: 2026-04-22
Context: Moving from a single IMX274 camera (Holoscan) setup to a dual-USB-camera setup (two UGREEN 2K cameras) so the VLA can reason about depth. This covers dataset generation, format conversion, video-codec issues, and training-time video-backend issues.

The issues are listed in the **order they were hit**, so you can see how each fix unblocked the next problem and identify where you might want to intervene differently in the future.

---

## 1. `ModuleNotFoundError: No module named 'lerobot'` when running the inference pipeline

### Symptom
```
File "/home/latticeapp/hsb-groot-robot/pipeline/linux_imx274_player.py", line 56, in _create_motor_bus
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
ModuleNotFoundError: No module named 'lerobot'
```

### Cause
`pipeline/linux_imx274_player.py` is meant to run **inside the Holoscan Docker container** because it needs both Holoscan (only installed in the container) and `lerobot` (installed via `install_robot_deps.sh` into `~/.docker-packages`, which is only on `PYTHONPATH` inside the container via `run.sh`).

The user was running it on the host, where `lerobot` is not importable.

### Fix
Run it inside the container:
```bash
./run.sh python pipeline/linux_imx274_player.py --camera-mode 1 ...
```
Also ensure `install_robot_deps.sh` has been run at least once.

### Caveat
`run.sh` fails on the host with `docker: command not found` on the Jetson because Docker isn't in PATH for the regular user. The fix was to `sudo -i` / run as root, OR run `bash install_robot_deps.sh` directly from a root shell already inside the container.

---

## 2. `docker: command not found` when trying to run `./run.sh bash install_robot_deps.sh`

### Cause
The user was **already inside the Docker container** (prompt showed `root@latticeapp`). `./run.sh` tries to re-docker-run, but `docker` is not available inside the container (we don't install Docker-in-Docker).

### Fix
Drop `./run.sh` and run the install script directly:
```bash
bash install_robot_deps.sh
```

### Caveat
This step is slow (~several minutes the first time) because it does a `pip install lerobot` + OpenCV + pyzmq into `~/.docker-packages`. Be patient — it looks like it's hung but it isn't.

---

## 3. `ValueError: Unrecognized processing class` when loading a trained checkpoint into the policy server

### Symptom
```
ValueError: Unrecognized processing class in /home/latticeapp/hsb-groot-robot/checkpoints/control_blue_only.
Can't instantiate a processor, a tokenizer, an image processor or a feature extractor for this model.
```

### Cause
We were pointing the server at the **top-level checkpoint folder**, e.g. `checkpoints/control_blue_only/`. In that folder, `processor_config.json` is nested in a `processor/` subfolder. `AutoProcessor.from_pretrained()` looks for it at the top level.

Individual numbered checkpoint folders (`checkpoint-2000/`, `checkpoint-4000/`…) have `processor_config.json` directly at their root — so those load correctly.

### Fix
Point the policy server at a numbered checkpoint, e.g.:
```bash
--model-path /home/latticeapp/hsb-groot-robot/checkpoints/control_blue_only/checkpoint-2000
```

### Caveat
When selecting a checkpoint to deploy, prefer the last one (or the one with lowest eval loss). Remember to pass the **checkpoint-XXXX** folder, not the parent.

---

## 4. USB camera detection / streaming check

### Problem
Two UGREEN USB cameras were plugged in. Needed to verify they were enumerated and produced video.

### Discovery
```bash
ls /dev/video*           # /dev/video0  /dev/video1  /dev/video2  /dev/video3
v4l2-ctl --list-devices   # two UGREEN Camera 2K devices → video0 and video2 are the capture nodes
```
`video1` and `video3` are the metadata / alt-interface nodes for the same physical cameras.

### Useful commands
- Grab one frame: `ffmpeg -f v4l2 -i /dev/video0 -frames:v 1 -update 1 /tmp/cam0.jpg`
- Live preview: `ffplay -f v4l2 -i /dev/video0`

### Caveat
The cameras default to **800×600 @ 20 fps YUYV** even though they're advertised as "2K". FFmpeg will helpfully log `The V4L2 driver changed the video from 2560x1440 to 800x600` — that's a driver-negotiated fallback. If you want higher res, you need to explicitly request it via `-video_size 1920x1080 -framerate 30` (but check `v4l2-ctl --list-formats-ext` first).

---

## 5. `ERROR: Missing modality.json in .../meta/ — run setup first.` after recording a dual-camera dataset

### Cause
GR00T's training pipeline requires `<dataset>/meta/modality.json` to describe which keys are video/state/action/language modalities. The original single-camera recording flow created this file as part of its setup. Our brand-new `recording/usb_dual_cam_record.py` did not.

### Fix
1. **Short term** — wrote the `modality.json` by hand for the existing `dual_cam_blue_only` dataset:
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
2. **Long term** — added a `_write_modality_json()` helper to `recording/usb_dual_cam_record.py` that runs in the `finally` block of `main()`, so every future recording writes it automatically.

### Caveat
The mapping between state/action chunks and joint indices **must match** what `so101_dual_usb_config.py` says, or GR00T will silently mis-index joints at train time. Right now: joints 0-4 are `single_arm` (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll) and joint 5 is `gripper`.

---

## 6. `ImportError: Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'` in the v3→v2 converter

### Cause
`training/convert_v3_to_v2.py` uses `pandas.read_parquet` → needs `pyarrow`. `pyarrow` is installed in the **`lerobot2` conda env**, but the user was in the plain `latticeapp@` shell.

### Fix
```bash
conda activate lerobot2
python training/convert_v3_to_v2.py datasets/dual_cam_blue_only
```

### Caveat
This is easy to forget. Consider adding a `conda run -n lerobot2` wrapper inside the converter, or an explicit env-check at the top of the script.

---

## 7. `FileNotFoundError: No such file or directory: '.../meta/episodes.jsonl'` at training start

### Cause
The `recording/usb_dual_cam_record.py` script (via the LeRobot library) writes datasets in **LeRobot v3.0 format**, which stores episode metadata in a single parquet file. GR00T's training code expects the older **v2.1 format**, which uses `episodes.jsonl` and `tasks.jsonl`.

### Fix
Run the converter **before training**:
```bash
conda activate lerobot2
python training/convert_v3_to_v2.py datasets/dual_cam_blue_only
```
This writes `episodes.jsonl`, `tasks.jsonl`, and rearranges the per-episode video paths so GR00T can find them.

### Caveat
- The converter is not idempotent — if you re-run it on an already-converted dataset you may end up with mismatched metadata. Keep a backup of the raw v3 dataset until you've confirmed training works.
- If we ever upgrade Isaac-GR00T to a version that natively supports v3, this conversion step should be deleted.

---

## 8. `NotImplementedError` inside `gr00t/utils/video_utils.py` during dataloading

### Symptom (training-time, worker process)
```
04/22/2026 16:11:36 - WARNING - Video backend 'torchcodec' is not available, falling back to 'pyav'.
...
File ".../gr00t/utils/video_utils.py", line 402, in get_frames_by_indices
    raise NotImplementedError
```

### Cause — this one is a chain of three bugs
1. GR00T's default video backend is `torchcodec`.
2. `torchcodec` (the version pinned in the Isaac-GR00T `.venv`) dynamically links against **FFmpeg 7** shared libraries. Ubuntu 24.04 / JetPack 6 ships **FFmpeg 6**, so `torchcodec`'s shared object fails to load → it's reported as "not available".
3. GR00T's automatic fallback order tries `pyav` next. In `get_frames_by_indices`, **there is no `pyav` branch** — it falls straight through to `raise NotImplementedError`.

### Fix — `training/launch_finetune_ffmpeg.py` wrapper
Created a thin wrapper that sits in front of `gr00t.experiment.launch_finetune`:
```python
# Override before GR00T imports anything.
from gr00t.configs.data import data_config as _data_config_mod
_data_config_mod.DataConfig.video_backend = "opencv"

# Also reorder the fallback list so opencv wins if anything bypasses the config.
from gr00t.utils import video_utils as _video_utils_mod
_video_utils_mod._BACKEND_FALLBACK_ORDER = ["opencv", "torchcodec", "decord", "ffmpeg", "pyav"]
```
Then `runpy.run_module("gr00t.experiment.launch_finetune", run_name="__main__")` forwards every CLI arg to the real launcher.

### Why not just install torchcodec properly?
- We'd need to either (a) rebuild `torchcodec` against FFmpeg 6, or (b) upgrade the system to FFmpeg 7, which breaks half of the OS packages on Ubuntu 24.04.
- `decord` has no ARM64 wheels on PyPI, so that isn't an option on the Jetson either.
- `pyav` works but isn't implemented in `get_frames_by_indices`.
- → `opencv` is the simplest working option and already ships with the system.

### Caveat
Any future Isaac-GR00T update may re-add a code path that imports `get_frames_by_indices` **by reference** (e.g. `from ...video_utils import get_frames_by_indices`). In that case our monkey-patch of the module-level symbol won't reach it — see issue #11 below for how we handle this.

---

## 9. Training "stuck" when using the `ffmpeg` backend

### Symptom
First attempt at fixing #8 forced `video_backend = "ffmpeg"`. Training got past the fallback warning but then sat at `0/2000` seemingly forever.

### Cause
Looking at `_extract_frames_ffmpeg` in `gr00t/utils/video_utils.py`: **it spawns a new `ffmpeg` subprocess per frame** (!) via `subprocess.run(...)`. With 16-step action chunks × multiple cams × 4 dataloader workers, this is thousands of `fork+exec` per step. It's not actually stuck — it's just that slow.

### Fix
Switched to `opencv` instead (single `VideoCapture` handle per clip). See `launch_finetune_ffmpeg.py` — despite the filename (kept for continuity) it now forces `opencv`.

### Caveat
The filename `launch_finetune_ffmpeg.py` is now misleading. TODO: rename to `launch_finetune_opencv.py` and update `finetune_dual_usb.sh`.

---

## 10. OpenCV can't decode the recorded videos — AV1 encoding on Jetson

### Symptom
```
[av1 @ 0x...] Missing Sequence Header. Your platform doesn't support hardware accelerated AV1 decoding.
ValueError: Unable to read frame at index 0
```

### Cause
- `recording/usb_dual_cam_record.py` (via LeRobot) asks FFmpeg for `h264_nvenc` encoding.
- On this particular Jetson + FFmpeg 6 build, `h264_nvenc` isn't available, so FFmpeg **silently falls back** to the first available codec in its auto-negotiation list — which turns out to be **AV1** (software libaom or svtav1).
- Jetson Thor has no hardware AV1 decoder exposed through FFmpeg, and the CPU AV1 decoder in use is missing sequence headers for the stream LeRobot wrote (likely a keyframe-interval / packaging issue).
- End result: some of the 60 recorded video files are AV1 that OpenCV cannot decode.

### Fix — `training/transcode_av1_to_h264.py`
Walks the dataset, probes each `.mp4` with `ffprobe`, and re-encodes any non-h264 ones to `libx264` (CPU-only, preset=fast, crf=23). Rewrites `info.json` so the LeRobot metadata reports `video.codec: h264`.

Run it once after each recording:
```bash
python3 training/transcode_av1_to_h264.py datasets/dual_cam_blue_only
```

### Real outcome
Of the 60 video files in `dual_cam_blue_only`:
- **57 were already h264** (most of the dataset was fine)
- **4 were AV1** (those were the ones crashing training)
- **1 leftover `.h264.mp4`** — temp file from an earlier failed run; cleaned up with `find . -name '*.h264.mp4' -delete`

### Caveat
- The real fix is to prevent AV1 output in the first place. The recording script should **explicitly pass `-c:v libx264` to FFmpeg** instead of trusting LeRobot's auto-selection. TODO in `recording/usb_dual_cam_record.py`.
- `libx264` CPU encoding is fine for our 30-episode datasets but won't scale to large datasets. If you ever record thousands of episodes, find a way to get `h264_nvenc` working (probably requires a newer FFmpeg built with CUDA 13 support).

---

## 11. OpenCV backend usable but **extremely slow** (each shard took minutes)

### Symptom
Training started (progress bar appeared) but the 4 dataloader workers sat at 99% CPU for 10+ minutes before step 1 happened. Actually stepping once took ~30 s with 0% GPU util.

### Cause
The upstream OpenCV code path in `get_frames_by_indices` does:
```python
for idx in indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)   # seek to nearest keyframe + decode forward
    ret, frame = cap.read()
```
`cap.set(CAP_PROP_POS_FRAMES)` does a **full keyframe seek every time**, then re-decodes from the keyframe. For the sequential indices that GR00T passes (`np.arange(actual_length)`), this means we decode from the start of the clip `N × (N+1) / 2` times instead of `N` times. O(N²) instead of O(N).

Plus there was a second latent bug: the opencv path returns `BGR`, but every other backend returns `RGB`. Training data would have been colour-shuffled.

### Fix — in-process monkey-patch (in `launch_finetune_ffmpeg.py`)
1. Replace `gr00t.utils.video_utils.get_frames_by_indices` with a fast version that:
   - Detects sequential indices and just calls `cap.read()` in order (no seek).
   - Only seeks when it has to jump backward or skip more than 30 frames.
   - Converts BGR → RGB so it matches every other backend.
2. **Also** patch `gr00t.data.dataset.lerobot_episode_loader.get_frames_by_indices` — because `lerobot_episode_loader.py` does `from gr00t.utils.video_utils import get_frames_by_indices`, creating a local binding that module-level reassignment doesn't reach. (This was easy to miss and a good example of a "from X import Y"-style gotcha.)

### Measured result
- Before patch: ~30 s / step, GPU util near 0%.
- After patch: **~2.3 s / step**, GPU util ~20–40%, ETA for 2000 steps ≈ 75 minutes.

### Caveat
- DataLoader workers on Linux default to `fork`, which means they inherit the monkey-patch. If PyTorch ever switches to `spawn` on this platform, workers will re-import the modules cleanly and our patch will vanish. If that happens, move the patch into a module that's imported **from inside every worker** (e.g. via a `worker_init_fn`).
- Any pull from upstream Isaac-GR00T that changes the signature of `get_frames_by_indices` will silently bypass the patch. Add a `inspect.signature` assertion to the wrapper if you want to catch that.

---

## 12. Thermal throttling after ~15 minutes of training

### Symptom
System reports thermal throttling; training slows down or the Jetson freezes.

### Cause
2000 steps × 2.3 s/step = **75 minutes of sustained GPU + CPU load** on a passively-cooled Jetson Thor dev kit. At roughly 40–60% GPU + 400% CPU on the dataloader workers, the SOC runs hot.

### Fixes / mitigations
1. Confirm max power mode: `sudo nvpmodel -q` (should be MAXN) and `sudo jetson_clocks --show`.
2. Physically: point a fan at the heatsink. This is the biggest lever.
3. Run detached so an SSH drop doesn't kill training:
   ```bash
   nohup bash training/finetune_dual_usb.sh dual_cam_blue_only dual_cam_blue_only > /tmp/train.log 2>&1 &
   ```
4. Let `save_steps=500` do its job — if it crashes after step 500 you can resume.

### Caveat
- The script currently uses `--save-steps 500`. You may want to drop this to `--save-steps 250` on a thermally unstable system so you lose less progress per crash.
- **`--resume` is not a valid CLI arg for `launch_finetune.py`**. Checkpoints are resumed from automatically if the output-dir already contains `checkpoint-XXXX/` folders — so to resume after a crash, just rerun the exact same command (no extra flag).

---

## Summary of new / modified files

| File | Purpose |
|---|---|
| `recording/usb_dual_cam_record.py` | New. Dual-USB-cam LeRobot recorder; also writes `meta/modality.json` at the end. |
| `training/so101_dual_usb_config.py` | New. Modality config for cam0 + cam1 + SO-101 arm + gripper. |
| `training/finetune_dual_usb.sh` | New. Launches training with the dual-cam config via the wrapper. |
| `training/launch_finetune_ffmpeg.py` | New. Wrapper around `launch_finetune.py` that forces `opencv` backend and monkey-patches the fast sequential reader. |
| `training/transcode_av1_to_h264.py` | New. Post-processes a recorded dataset to re-encode any AV1 files as h264, and updates `info.json`. |
| `training/convert_v3_to_v2.py` | Existing. Converts LeRobot v3 datasets to v2.1 (the format GR00T expects). Needs `conda activate lerobot2`. |

## Recommended pipeline (end-to-end)

```bash
# 1. Record dataset (writes v3 + modality.json)
python3 recording/usb_dual_cam_record.py --repo-id dual_cam_blue_only --task "Move the blue dice" ...

# 2. Re-encode any AV1 videos to h264 (idempotent, safe to always run)
python3 training/transcode_av1_to_h264.py datasets/dual_cam_blue_only

# 3. Convert v3 → v2.1 (requires lerobot2 conda env for pyarrow)
conda activate lerobot2
python training/convert_v3_to_v2.py datasets/dual_cam_blue_only

# 4. Train (in any shell; wrapper handles venv + LD_LIBRARY_PATH)
bash training/finetune_dual_usb.sh dual_cam_blue_only dual_cam_blue_only 2>&1 | tee /tmp/train.log

# 5. (Other terminal) Watch progress without warning spam
tail -f /tmp/train.log | tr '\r' '\n' | grep --line-buffered "it/s"

# 6. If it dies mid-training, just re-run step 4 — resumes from last checkpoint automatically.
```

## Possible points of intervention (if you want to do things differently next time)

1. **Swap opencv for a real video backend.** If someone can get `torchcodec` building against system FFmpeg 6, or can get `decord` compiled for ARM64, all of #8, #9, #11 just… go away.
2. **Force `libx264` at record time.** Fixing this one line in the recorder makes the transcode step in #10 unnecessary.
3. **Skip the v3→v2 converter** by either pinning an older LeRobot (writes v2.1 directly) or upgrading Isaac-GR00T to one that reads v3 natively.
4. **Inline `modality.json` / schema checks in the recorder** so a bad recording fails immediately instead of at training time.
5. **Active cooling on the Jetson.** Probably the cheapest and highest-impact change for reliability.
