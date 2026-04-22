"""Convert a locally recorded LeRobot v3.0 dataset to the v2.1 format
that GR00T's LeRobotEpisodeLoader expects.

Changes made:
  - data/chunk-NNN/file-NNN.parquet  →  data/chunk-NNN/episode_NNNNNN.parquet
  - videos/<key>/chunk-NNN/file-NNN.mp4  →  videos/chunk-NNN/<key>/episode_NNNNNN.mp4
  - info.json  data_path / video_path fields updated to v2.1 templates
  - meta/episodes.jsonl  and  meta/tasks.jsonl  generated if missing

The original dataset is left in a <name>_v30 backup folder. The converted
dataset overwrites the original path so all existing scripts still point to it.

Usage (from any directory):
    conda activate lerobot2
    python ~/hsb-groot-robot/training/convert_v3_to_v2.py <dataset_path>

Example:
    python ~/hsb-groot-robot/training/convert_v3_to_v2.py \
        ~/hsb-groot-robot/datasets/control_blue_only
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ── v2.1 path templates (what GR00T expects) ──────────────────────────────────
V2_DATA_PATH  = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
V2_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def load_episode_records(root: Path) -> list[dict]:
    ep_dir = root / "meta" / "episodes"
    frames = [pd.read_parquet(f) for f in sorted(ep_dir.glob("**/*.parquet"))]
    ep_df = pd.concat(frames).sort_values("episode_index").reset_index(drop=True)
    return ep_df.to_dict("records")


def write_jsonl(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def convert_meta(root: Path, new_root: Path, episode_records: list[dict], video_keys: list[str], chunks_size: int):
    """Write info.json, tasks.jsonl, episodes.jsonl to new_root."""
    info = json.loads((root / "meta" / "info.json").read_text())

    # Update path templates to v2.1
    info["codebase_version"] = "v2.1"
    info["data_path"]  = V2_DATA_PATH
    info["video_path"] = V2_VIDEO_PATH
    info["total_episodes"] = len(episode_records)
    info["total_chunks"]   = max(1, -(-len(episode_records) // chunks_size))  # ceil div

    (new_root / "meta").mkdir(parents=True, exist_ok=True)
    (new_root / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    # tasks.jsonl
    tasks_parquet = pd.read_parquet(root / "meta" / "tasks.parquet")
    task_records = [
        {"task_index": int(row["task_index"]), "task": str(task)}
        for task, row in tasks_parquet.iterrows()
    ]
    write_jsonl(new_root / "meta" / "tasks.jsonl", task_records)

    # episodes.jsonl
    ep_records_out = []
    for rec in episode_records:
        tasks_val = rec["tasks"]
        tasks_list = tasks_val.tolist() if isinstance(tasks_val, np.ndarray) else list(tasks_val)
        ep_records_out.append({
            "episode_index": int(rec["episode_index"]),
            "tasks":         tasks_list,
            "length":        int(rec["length"]),
        })
    write_jsonl(new_root / "meta" / "episodes.jsonl", ep_records_out)

    # copy stats.json if present
    stats_src = root / "meta" / "stats.json"
    if stats_src.exists():
        shutil.copy2(stats_src, new_root / "meta" / "stats.json")

    # copy modality.json if present
    mod_src = root / "meta" / "modality.json"
    if mod_src.exists():
        shutil.copy2(mod_src, new_root / "meta" / "modality.json")

    print(f"  meta/ written ({len(episode_records)} episodes, {len(task_records)} tasks)")


def convert_data(root: Path, new_root: Path, episode_records: list[dict], chunks_size: int):
    """Split multi-episode parquet files into one file per episode."""
    info = json.loads((root / "meta" / "info.json").read_text())
    v3_data_path = info["data_path"]  # e.g. data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet

    print(f"  Converting data parquets ({len(episode_records)} episodes)...")
    for rec in episode_records:
        ep_idx    = int(rec["episode_index"])
        chunk_idx = int(rec["data/chunk_index"])
        file_idx  = int(rec["data/file_index"])

        src = root / v3_data_path.format(chunk_index=chunk_idx, file_index=file_idx)
        df  = pd.read_parquet(src)
        df_ep = df[df["episode_index"] == ep_idx].reset_index(drop=True)

        ep_chunk = ep_idx // chunks_size
        dst = new_root / V2_DATA_PATH.format(episode_chunk=ep_chunk, episode_index=ep_idx)
        dst.parent.mkdir(parents=True, exist_ok=True)
        df_ep.to_parquet(dst, index=False)

    print(f"  Data conversion done.")


def convert_videos(root: Path, new_root: Path, episode_records: list[dict], video_keys: list[str], chunks_size: int):
    """Reorganise video files from v3 layout to v2.1 layout."""
    info = json.loads((root / "meta" / "info.json").read_text())
    v3_video_path = info["video_path"]  # videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4

    print(f"  Converting videos ({len(video_keys)} camera(s) × {len(episode_records)} episodes)...")
    for rec in episode_records:
        ep_idx = int(rec["episode_index"])
        ep_chunk = ep_idx // chunks_size

        for vkey in video_keys:
            chunk_col = f"videos/{vkey}/chunk_index"
            file_col  = f"videos/{vkey}/file_index"
            chunk_idx = int(rec.get(chunk_col, rec.get("data/chunk_index", 0)))
            file_idx  = int(rec.get(file_col,  rec.get("data/file_index",  0)))

            # v3 source — try the pattern from info.json first
            try:
                src = root / v3_video_path.format(
                    video_key=vkey, chunk_index=chunk_idx, file_index=file_idx
                )
            except KeyError:
                src = root / f"videos/{vkey}/chunk-{chunk_idx:03d}/file-{file_idx:03d}.mp4"

            if not src.exists():
                print(f"  WARN: video not found: {src}")
                continue

            dst = new_root / V2_VIDEO_PATH.format(
                episode_chunk=ep_chunk, video_key=vkey, episode_index=ep_idx
            )
            dst.parent.mkdir(parents=True, exist_ok=True)

            # Extract single-episode clip if the source file has multiple episodes
            # (for now copy — ffmpeg trim can be added later if needed)
            shutil.copy2(src, dst)

    print("  Video conversion done.")


def convert(dataset_path: Path):
    root = dataset_path.resolve()
    if not root.exists():
        print(f"ERROR: dataset not found: {root}")
        sys.exit(1)

    info = json.loads((root / "meta" / "info.json").read_text())
    version = info.get("codebase_version", "")
    if not version.startswith("v3"):
        print(f"Dataset is already version '{version}', not v3.0 — nothing to do.")
        sys.exit(0)

    chunks_size = info.get("chunks_size", 1000)
    video_keys  = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    episode_records = load_episode_records(root)

    # Work into a sibling _v21 folder, then swap
    new_root    = root.parent / f"{root.name}_v21"
    backup_root = root.parent / f"{root.name}_v30"

    if new_root.exists():
        shutil.rmtree(new_root)

    print(f"\nConverting: {root}")
    print(f"  Version     : {version} → v2.1")
    print(f"  Episodes    : {len(episode_records)}")
    print(f"  Video keys  : {video_keys}")
    print(f"  Chunks size : {chunks_size}")
    print()

    convert_meta(root, new_root, episode_records, video_keys, chunks_size)
    convert_data(root, new_root, episode_records, chunks_size)
    convert_videos(root, new_root, episode_records, video_keys, chunks_size)

    # Swap: original → _v30 backup, converted → original path
    print(f"\n  Backing up original → {backup_root.name}")
    if backup_root.exists():
        shutil.rmtree(backup_root)
    shutil.move(str(root), str(backup_root))

    print(f"  Moving converted dataset → {root.name}")
    shutil.move(str(new_root), str(root))

    print(f"\nDone. Dataset at: {root}")
    print(f"Original backed up at: {backup_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LeRobot v3.0 dataset to v2.1 for GR00T.")
    parser.add_argument("dataset_path", type=Path, help="Path to the v3.0 dataset folder")
    args = parser.parse_args()
    convert(args.dataset_path)
