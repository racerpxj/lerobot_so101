#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Export a LeRobot dataset to readable formats (CSV/JSON + extracted video frames).
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import cv2


def _load_parquet(path: Path):
    try:
        import pandas as pd  # type: ignore

        return pd.read_parquet(path)
    except Exception:
        import pyarrow.parquet as pq  # type: ignore

        return pq.read_table(path).to_pandas()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        json.dump(rows, f, indent=2, default=str)


def _extract_video(video_path: Path, output_dir: Path, max_frames: int) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out_path = output_dir / f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(out_path), frame)
        frame_idx += 1
        if max_frames > 0 and frame_idx >= max_frames:
            break
    cap.release()
    return frame_idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a LeRobot dataset to readable formats.")
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Dataset root (e.g. ~/.cache/huggingface/lerobot/local/eval_so101_pi05_debug)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for readable exports",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=0,
        help="Maximum rows to export from parquet (0 = all)",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Maximum frames to export from each video (0 = all)",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser()
    output_root = Path(args.output_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    data_dir = dataset_root / "data"
    videos_dir = dataset_root / "videos"
    meta_dir = dataset_root / "meta"

    # Export parquet data
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    for parquet_path in parquet_files:
        rel = parquet_path.relative_to(dataset_root)
        out_base = output_root / rel.parent
        out_base.mkdir(parents=True, exist_ok=True)

        df = _load_parquet(parquet_path)
        if args.max_rows > 0:
            df = df.head(args.max_rows)
        rows = df.to_dict(orient="records")

        _write_csv(out_base / f"{parquet_path.stem}.csv", rows)
        _write_json(out_base / f"{parquet_path.stem}.json", rows)

    # Export meta files
    if meta_dir.exists():
        meta_out = output_root / "meta"
        meta_out.mkdir(parents=True, exist_ok=True)
        for meta_file in meta_dir.rglob("*"):
            if meta_file.is_file():
                target = meta_out / meta_file.relative_to(meta_dir)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(meta_file.read_bytes())

    # Extract videos
    if videos_dir.exists():
        for video_path in sorted(videos_dir.rglob("*.mp4")):
            rel = video_path.relative_to(dataset_root)
            out_dir = output_root / rel.with_suffix("")
            count = _extract_video(video_path, out_dir, args.max_frames)
            print(f"Extracted {count} frames from {video_path} -> {out_dir}")

    print(f"Export complete: {output_root}")


if __name__ == "__main__":
    main()
