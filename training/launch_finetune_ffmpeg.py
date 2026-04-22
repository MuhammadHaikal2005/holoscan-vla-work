"""Thin wrapper around gr00t.experiment.launch_finetune that forces the
video backend to 'opencv' instead of the default 'torchcodec'.

Needed because torchcodec requires FFmpeg 7, but Ubuntu 24.04 / JetPack 6
ships FFmpeg 6 — torchcodec fails to load. Its fallback 'pyav' is NOT
implemented in gr00t.utils.video_utils.get_frames_by_indices, and 'ffmpeg'
spawns a subprocess PER FRAME which makes training unusably slow.

OpenCV uses a single VideoCapture handle per episode and works with the
system FFmpeg 6 libraries out of the box.

Usage is IDENTICAL to gr00t/experiment/launch_finetune.py — all CLI args
are forwarded untouched.
"""
import os
import runpy
import sys

# ── 1. Override the default video backend BEFORE gr00t imports anything ────
from gr00t.configs.data import data_config as _data_config_mod

_data_config_mod.DataConfig.video_backend = "opencv"

# ── 2. Also patch the fallback order so pyav/ffmpeg are de-prioritised ─────
#    in case any code path bypasses our config.
from gr00t.utils import video_utils as _video_utils_mod

_video_utils_mod._BACKEND_FALLBACK_ORDER = ["opencv", "torchcodec", "decord", "ffmpeg", "pyav"]

# ── 3. Patch get_frames_by_indices with a fast sequential opencv reader ────
#    The upstream opencv path calls cap.set(CAP_PROP_POS_FRAMES, idx) before
#    each frame, triggering a keyframe seek + forward decode per frame —
#    orders of magnitude slower than needed for sequential reads.
#    Also upstream returns BGR for opencv but RGB for every other backend,
#    so we convert BGR→RGB here for consistency with torchcodec/decord/ffmpeg.
import numpy as _np
import cv2 as _cv2

_original_get_frames_by_indices = _video_utils_mod.get_frames_by_indices


def _fast_get_frames_by_indices(
    video_path, indices, video_backend="ffmpeg", video_backend_kwargs={},
):
    resolved = _video_utils_mod.resolve_backend(video_path, video_backend)
    if resolved != "opencv":
        return _original_get_frames_by_indices(
            video_path, indices, video_backend=video_backend,
            video_backend_kwargs=video_backend_kwargs,
        )

    indices = list(indices)
    idx_to_pos = {int(idx): pos for pos, idx in enumerate(indices)}
    wanted = sorted(idx_to_pos.keys())
    frames = [None] * len(indices)

    cap = _cv2.VideoCapture(video_path, **video_backend_kwargs)
    try:
        current = 0
        wanted_iter = iter(wanted)
        next_wanted = next(wanted_iter, None)
        while next_wanted is not None:
            # If we'd need to go backwards, or skip far ahead, seek.
            # For sequential reads (next_wanted == current) we just keep reading.
            if next_wanted < current or next_wanted - current > 30:
                cap.set(_cv2.CAP_PROP_POS_FRAMES, next_wanted)
                current = next_wanted
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Unable to read frame at index {current} in {video_path}")
            if current == next_wanted:
                frame_rgb = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
                frames[idx_to_pos[current]] = frame_rgb
                next_wanted = next(wanted_iter, None)
            current += 1
    finally:
        cap.release()

    return _np.array(frames)


_video_utils_mod.get_frames_by_indices = _fast_get_frames_by_indices

# Also patch the copy that lerobot_episode_loader bound via `from ... import`.
from gr00t.data.dataset import lerobot_episode_loader as _lel_mod
_lel_mod.get_frames_by_indices = _fast_get_frames_by_indices

# ── 3. Delegate to the real launcher ──────────────────────────────────────
# Re-exec launch_finetune.py with the same argv, running it as __main__.
_LAUNCH_MODULE = "gr00t.experiment.launch_finetune"
print(f"[launch_finetune_ffmpeg] video_backend forced to 'opencv'")
runpy.run_module(_LAUNCH_MODULE, run_name="__main__", alter_sys=True)
