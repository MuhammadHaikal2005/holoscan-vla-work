# Dataloader Workers, CPU Overload, and the OpenCV Monkey-Patch — Explained

**Date:** 2026-04-23  
**Context:** During training on both the Thor and the 3090, several confusing behaviours appeared — CPU usage well above 100%, the terminal showing "Caching shard..." for several minutes with no progress bar, and an earlier session using a monkey-patch to swap the video backend. This note explains the "why" behind all of it.

---

## Why CPU usage goes above 100%

The number shown by `top`, `htop`, or `ps` is **per-core CPU usage**. A value of 100% means one full CPU core is saturated. A value of 189% means the process is using 1.89 cores simultaneously — which is normal and expected for multi-threaded work.

On the Jetson AGX Thor (14-core CPU) and the 3090 PC (also multi-core), the training process spawns several parallel workers. Each worker runs on its own core. When you see four workers each at ~185%, that is:

```
4 workers × 185% ≈ 7.4 cores fully saturated
```

This is the intended behaviour — it is the dataloader working as fast as it can. The workers are not "stuck" or "broken". They are decoding video frames from disk and pre-processing them into tensors as quickly as the hardware allows.

The reason this looked alarming on the Thor is that 7+ fully loaded cores draws significant CPU power, which pushed total system draw close to the PSU's delivery limit and triggered the over-current notifications.

---

## What the dataloader workers actually do

PyTorch's `DataLoader` uses a **producer–consumer architecture**:

```
Disk / Storage
      │
      ▼
┌─────────────────────────────────────────┐
│         Dataloader Workers (CPU)        │
│  Worker 0 │ Worker 1 │ Worker 2 │ ...  │
│                                         │
│  1. Read episode parquet (state/action) │
│  2. Open video file                     │
│  3. Decode frames with torchcodec       │
│  4. Resize + normalise to tensors       │
│  5. Package into a batch dict           │
└─────────────────────────────────────────┘
      │
      ▼  (placed in a shared memory queue)
┌─────────────────────────────────────────┐
│          Main Process (GPU)             │
│                                         │
│  1. Pull batch from queue               │
│  2. Copy tensors to VRAM                │
│  3. Model forward pass                  │
│  4. Compute loss                        │
│  5. Backward pass + optimizer step      │
└─────────────────────────────────────────┘
```

The workers run **entirely on CPU**. The GPU only ever sees the finished tensors — it has no involvement in video decoding or data preparation. This is why the choice of video backend (torchcodec, opencv, pyav) has no impact on training quality — it only affects how fast the CPU workers can feed the GPU.

The `--dataloader-num-workers` argument controls how many of these CPU processes run in parallel. More workers = more CPU cores used = batches ready faster = GPU stays busier.

---

## What "Caching shard" means and why the progress bar is delayed

GR00T does not load episodes one at a time during training. Instead it groups the dataset into **shards** — large contiguous blocks of pre-processed samples that a worker can iterate through sequentially without constantly reopening files.

Before any training step can happen, each worker must load and cache its assigned shard fully into memory. The sequence is:

```
1. Training starts
2. Each worker receives a list of assigned shards
3. Each worker opens its first shard:
   - reads all episode parquets
   - decodes all video frames for those episodes
   - applies all image transforms (resize, crop, normalise)
   - stores the result in a shared memory buffer
4. Only after ALL workers have finished caching does
   the main process start pulling batches
5. The progress bar appears at this point
```

The "Caching shard..." messages are printed at the start of step 3. If the dataset has a lot of video (especially with two cameras), step 3 takes several minutes. During this entire time the progress bar sits at `0/2000` and the terminal looks completely frozen. This is **normal and expected** — the workers are working, just not producing training steps yet.

The wait messages like:
```
Rank 0, Worker 1: Wait for shard 9 in dataset 0 in 94.29 seconds
```
mean one worker has finished its current shard and is waiting for the background thread to finish pre-caching the *next* shard. GR00T caches shards one step ahead so there is no pause between shards during training.

---

## Why `>100%` CPU during shard caching specifically

Shard caching is the heaviest CPU phase of training. All workers run simultaneously on separate cores, each decoding video at full speed. With 4 workers on a 2-camera dataset:

```
4 workers × 2 cameras × N frames × decode + resize + normalise
```

Each `VideoCapture.read()` or `torchcodec` decode call uses one core. Torchcodec can use multiple threads per call, which is why individual workers sometimes show >100% (e.g. 185% = one worker using 1.85 cores internally via its own thread pool).

Once caching is done and training enters the steady state, each worker only needs to serve one shard at a time from a pre-built memory buffer — mostly memory reads, no video decoding. CPU usage drops dramatically and GPU becomes the bottleneck.

---

## The opencv monkey-patch — what it was and why it existed

### The problem chain

During the very first training attempt on the Thor:

1. **torchcodec** (GR00T's default video backend) failed to load because the version installed in Isaac-GR00T's venv was compiled against **FFmpeg 7** (`libavfilter.so.10`), but the Jetson's Ubuntu 24.04 installation only has **FFmpeg 6** (`libavfilter.so.9`). The library uses versioned ELF symbols (`@LIBAVCODEC_61`, etc.) and calls functions that only exist in FFmpeg 7 — so even a symlink from `.so.10` to `.so.9` could not fix it.

2. GR00T automatically **fell back to pyav** when torchcodec failed to load. But the pyav branch in GR00T's `get_frames_by_indices` function ended with `raise NotImplementedError`. It was never implemented.

3. The next fallback was **opencv**. OpenCV could actually open and read the video files, but GR00T's opencv path had two bugs:
   - It called `cap.set(CAP_PROP_POS_FRAMES, idx)` before every single frame read. This does a **full keyframe seek** each time, then decodes forward from that keyframe to the target frame. For sequential indices `[0, 1, 2, 3, …, N]` this means decoding frames `0` + `0,1` + `0,1,2` + … — O(N²) work instead of O(N).
   - It returned frames in **BGR** colour order (OpenCV's default) rather than RGB, meaning training data would have had red and blue channels swapped.

4. Training appeared completely frozen at `0/2000` — not because nothing was happening, but because the O(N²) decoding made each shard take 10+ minutes instead of ~30 seconds.

### The monkey-patch

Rather than modifying Isaac-GR00T's source files (which would break on `git pull`), a thin wrapper script (`training/launch_finetune_ffmpeg.py`) was written that ran *before* GR00T imported anything and patched the functions in memory:

```python
# 1. Force video backend to opencv before any GR00T code runs
from gr00t.configs.data import data_config as _dc
_dc.DataConfig.video_backend = "opencv"

# 2. Replace get_frames_by_indices with a fast sequential version
def _fast_get_frames(video_path, indices, ...):
    cap = cv2.VideoCapture(video_path)
    frames = []
    prev_idx = -1
    for idx in indices:
        # Only seek if we need to jump backward or skip many frames
        if idx != prev_idx + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # fix BGR→RGB
        prev_idx = idx
    return np.stack(frames)

# 3. Patch both locations where the function is bound
import gr00t.utils.video_utils as _vu
import gr00t.data.dataset.lerobot_episode_loader as _lel
_vu.get_frames_by_indices = _fast_get_frames
_lel.get_frames_by_indices = _fast_get_frames  # needed because of "from X import Y" binding
```

The double-patch (both `video_utils` and `lerobot_episode_loader`) was necessary because `lerobot_episode_loader.py` imports the function with `from gr00t.utils.video_utils import get_frames_by_indices` — which creates a **local binding** in that module's namespace. Patching only `video_utils.get_frames_by_indices` would leave `lerobot_episode_loader`'s local copy pointing to the old function.

### Why the monkey-patch was removed

The proper fix was to rebuild torchcodec from source against the system's FFmpeg 6. The cmake build script detects the installed FFmpeg version via `pkg-config` (libavcodec 60.x → FFmpeg 6) and produces `libtorchcodec_core6.so` which links against the correct `libavfilter.so.9`.

Once this source-built wheel was installed, torchcodec loaded correctly and GR00T used it directly — no wrapper, no patching, no opencv fallback, no BGR/RGB bug. The monkey-patch wrapper (`launch_finetune_ffmpeg.py`) became dead code and the training scripts were updated to call `launch_finetune.py` directly.

The source-built wheel is cached at `Isaac-GR00T/wheels/torchcodec-0.10.0a0-cp312-cp312-linux_aarch64.whl`. Both finetune scripts check for `libtorchcodec_core6.so` at startup and reinstall from the cache (~3 seconds) if `uv sync` has ever reset it to the PyPI FFmpeg-7 wheel.

---

## Summary of the dataloader + video backend stack

```
finetune_dual_usb.sh
    └── launch_finetune.py
            └── HuggingFace Trainer
                    └── ShardedMixtureDataset
                            └── DataLoader (--dataloader-num-workers N)
                                    └── Worker processes (CPU, N cores)
                                            └── LeRobotEpisodeLoader
                                                    └── get_frames_by_indices
                                                            └── torchcodec  ← video decode (CPU)
                                                                    ↓
                                                            RGB float tensors
                                                                    ↓
                                                    Model forward pass (GPU)
```

Everything above the GPU arrow runs on CPU. The video backend only affects how fast the CPU workers can decode frames. On the Thor (torchcodec, FFmpeg 6), the shard caching took 5–10 minutes. On the 3090 (torchcodec, FFmpeg via PyPI), it was comparable. The GPU training loop itself is identical regardless of backend.
