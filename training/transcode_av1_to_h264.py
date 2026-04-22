"""Transcode every AV1-encoded video in a LeRobot dataset to h264 (libx264).

Recordings made with vcodec=h264_nvenc can silently fall back to libsvtav1
on some systems, producing AV1 files that OpenCV cannot decode on Jetson
(no AV1 hardware decoder). This script re-encodes every .mp4 in the
dataset's videos/ tree to h264 yuv420p, in-place.

Usage (from any directory, lerobot2 env not required):
    python training/transcode_av1_to_h264.py datasets/dual_cam_blue_only

Safe to re-run — already-h264 files are skipped automatically.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def probe_codec(video_path: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error",
             "-select_streams", "v:0",
             "-show_entries", "stream=codec_name",
             "-of", "default=noprint_wrappers=1:nokey=1",
             str(video_path)],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return out or None
    except subprocess.CalledProcessError:
        return None


def transcode_one(src: Path) -> bool:
    """Transcode src in place to h264. Returns True on success."""
    tmp = src.with_suffix(".h264.mp4")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "23",
        "-an",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"    FAILED: {src}  ({e})")
        if tmp.exists():
            tmp.unlink()
        return False

    shutil.move(str(tmp), str(src))
    return True


def fix_info_json(dataset_root: Path) -> None:
    """Update info.json so its codec metadata matches reality."""
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        return
    info = json.loads(info_path.read_text())
    changed = False
    for feat in info.get("features", {}).values():
        vinfo = feat.get("info")
        if vinfo and vinfo.get("video.codec") != "h264":
            vinfo["video.codec"] = "h264"
            changed = True
    if changed:
        info_path.write_text(json.dumps(info, indent=4))
        print(f"  Updated {info_path} → video.codec = h264")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_path", type=Path)
    args = ap.parse_args()

    root = args.dataset_path.resolve()
    if not root.is_dir():
        print(f"ERROR: not a directory: {root}")
        sys.exit(1)

    videos_dir = root / "videos"
    if not videos_dir.is_dir():
        print(f"ERROR: no videos/ folder at {root}")
        sys.exit(1)

    all_videos = sorted(videos_dir.rglob("*.mp4"))
    print(f"Found {len(all_videos)} video file(s) under {videos_dir}")
    print()

    n_transcoded = n_skipped = n_failed = 0
    for i, vid in enumerate(all_videos, 1):
        codec = probe_codec(vid)
        rel = vid.relative_to(root)
        if codec == "h264":
            n_skipped += 1
            print(f"[{i}/{len(all_videos)}] SKIP  ({codec}):  {rel}")
            continue
        print(f"[{i}/{len(all_videos)}] TRANS ({codec} → h264):  {rel}")
        if transcode_one(vid):
            n_transcoded += 1
        else:
            n_failed += 1

    print()
    print(f"  transcoded: {n_transcoded}")
    print(f"  skipped   : {n_skipped}")
    print(f"  failed    : {n_failed}")

    fix_info_json(root)

    print("\nDone.")


if __name__ == "__main__":
    main()
