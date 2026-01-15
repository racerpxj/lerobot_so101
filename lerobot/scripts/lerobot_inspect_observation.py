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
Inspect what SO101 follower can observe and what formats are returned.

Example:
```bash
python /Users/hanyumo/Downloads/lerobot-main/src/lerobot/scripts/lerobot_inspect_observation.py \
  --port=/dev/tty.usbmodem5AB01587521 \
  --id=black \
  --front_index=1 \
  --wrist_index=0 \
  --width=1920 \
  --height=1080 \
  --front_fps=30 \
  --wrist_fps=5 \
  --num_frames=3 \
  --interval_s=0.5
```
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import Any

import cv2
import numpy as np
import torch

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.utils import init_logging


def _describe_value(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return _describe_array(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return _describe_array(value)
    return f"type={type(value).__name__} value={value}"


def _describe_array(array: np.ndarray) -> str:
    if array.size == 0:
        return f"shape={array.shape} dtype={array.dtype} empty"
    min_val = float(array.min())
    max_val = float(array.max())
    return f"shape={array.shape} dtype={array.dtype} min={min_val:.4f} max={max_val:.4f}"


def _open_camera(index: int, width: int, height: int, fps: float, label: str) -> cv2.VideoCapture:
    cam = cv2.VideoCapture(index)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    cam.set(cv2.CAP_PROP_FPS, float(fps))
    if not cam.isOpened():
        raise RuntimeError(f"Failed to open {label} camera (index={index}).")
    return cam


def _read_camera(cam: cv2.VideoCapture, label: str) -> np.ndarray | None:
    ok, frame = cam.read()
    if not ok:
        print(f"⚠️  {label} camera read failed.")
        return None
    return frame


def _frame_to_base64_jpg(frame: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG.")
    return base64.b64encode(buffer).decode("ascii")


def _save_frame(output_dir: str, obs: dict[str, Any], front_frame, wrist_frame) -> None:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    frame_dir = os.path.join(output_dir, f"frame_{timestamp}")
    os.makedirs(frame_dir, exist_ok=True)

    joint_order = [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ]
    state = np.array([float(obs.get(joint, 0.0)) for joint in joint_order], dtype=np.float32)

    np.savez(
        os.path.join(frame_dir, "observation.npz"),
        state=state,
        front_image=front_frame,
        wrist_image=wrist_frame,
    )

    np.save(os.path.join(frame_dir, "state.npy"), state)
    if front_frame is not None:
        np.save(os.path.join(frame_dir, "front_image.npy"), front_frame)
    if wrist_frame is not None:
        np.save(os.path.join(frame_dir, "wrist_image.npy"), wrist_frame)

    with open(os.path.join(frame_dir, "observation.json"), "w") as f:
        json.dump(obs, f, indent=2, default=str)

    if front_frame is not None:
        cv2.imwrite(os.path.join(frame_dir, "front.jpg"), front_frame)
    if wrist_frame is not None:
        cv2.imwrite(os.path.join(frame_dir, "wrist.jpg"), wrist_frame)

    print(f"📦 Saved frame data to {frame_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect SO101 observation and camera data formats.")
    parser.add_argument("--port", required=True, help="Serial port for SO101 follower")
    parser.add_argument("--id", default="black", help="Robot id")
    parser.add_argument("--front_index", type=int, default=1, help="OpenCV index for front camera")
    parser.add_argument("--wrist_index", type=int, default=0, help="OpenCV index for wrist camera")
    parser.add_argument("--width", type=int, default=1920, help="Camera width")
    parser.add_argument("--height", type=int, default=1080, help="Camera height")
    parser.add_argument("--front_fps", type=float, default=30.0, help="Front camera FPS")
    parser.add_argument("--wrist_fps", type=float, default=5.0, help="Wrist camera FPS")
    parser.add_argument("--num_frames", type=int, default=3, help="Number of frames to print")
    parser.add_argument("--interval_s", type=float, default=0.5, help="Seconds between frames")
    parser.add_argument("--calibrate", action="store_true", help="Run calibration on connect")
    parser.add_argument("--output_dir", default="", help="Directory to save observations/images")
    args = parser.parse_args()

    init_logging()

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id, cameras={}))
    front_cam = None
    wrist_cam = None

    try:
        robot.connect(calibrate=args.calibrate)

        front_cam = _open_camera(args.front_index, args.width, args.height, args.front_fps, "front")
        wrist_cam = _open_camera(args.wrist_index, args.width, args.height, args.wrist_fps, "wrist")

        print("Cameras:")
        print(
            f"  - front: index={args.front_index} {args.width}x{args.height} @ {args.front_fps} FPS"
        )
        print(
            f"  - wrist: index={args.wrist_index} {args.width}x{args.height} @ {args.wrist_fps} FPS"
        )

        for i in range(args.num_frames):
            obs = robot.get_observation()
            print(f"\nObservation #{i + 1}:")
            for key in sorted(obs.keys()):
                print(f"  - {key}: {_describe_value(obs[key])}")

            front_frame = _read_camera(front_cam, "front")
            wrist_frame = _read_camera(wrist_cam, "wrist")

            if front_frame is not None:
                print(f"  - front_image: {_describe_value(front_frame)}")
            if wrist_frame is not None:
                print(f"  - wrist_image: {_describe_value(wrist_frame)}")

            if args.output_dir:
                _save_frame(args.output_dir, obs, front_frame, wrist_frame)

            if i < args.num_frames - 1:
                time.sleep(args.interval_s)
    finally:
        if front_cam:
            front_cam.release()
        if wrist_cam:
            wrist_cam.release()
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
