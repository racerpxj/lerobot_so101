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
Send a saved observation frame to the remote policy server and print actions.

Example:
```bash
python /Users/hanyumo/Downloads/lerobot-main/src/lerobot/scripts/lerobot_remote_policy_frame_test.py \
  --frame_dir=/Users/hanyumo/Downloads/lerobot-main/outputs/so101_inspect/frame_20251226_170836_125277 \
  --server_address=127.0.0.1:5555 \
  --policy_type=pi05 \
  --policy_path=/root/autodl-tmp/eval_pi05_so101_pickplace_front_wrist_camera_merge_all \
  --policy_device=cuda \
  --task="pick and place"
```
"""

import argparse
import json
import io
import os
import pickle  # nosec
import sys
import time
import types
from typing import Any

import cv2
import grpc
import numpy as np
import torch

from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import send_bytes_in_chunks
from lerobot.utils.constants import OBS_STR


def _ensure_protobuf_runtime_version() -> None:
    """Ensure protobuf runtime version is available."""
    try:
        from google.protobuf import runtime_version  # noqa: F401
        return
    except Exception:
        module_name = "google.protobuf.runtime_version"
        if module_name in sys.modules:
            return
        runtime_version = types.ModuleType(module_name)

        class _Domain:
            PUBLIC = 0

        def _validate(*_args, **_kwargs) -> None:
            return None

        runtime_version.Domain = _Domain
        runtime_version.ValidateProtobufRuntimeVersion = _validate
        sys.modules[module_name] = runtime_version


_ensure_protobuf_runtime_version()


def _load_obs(frame_dir: str) -> dict[str, Any]:
    pkl_path = os.path.join(frame_dir, "observation.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Missing observation.pkl in {frame_dir}")
    with open(pkl_path, "rb") as f:
        obs = pickle.load(f)  # nosec
    if not isinstance(obs, dict):
        raise ValueError("observation.pkl must contain a dict.")
    return obs


def _load_image(frame_dir: str, name: str) -> np.ndarray | None:
    path = os.path.join(frame_dir, f"{name}.jpg")
    if not os.path.exists(path):
        return None
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def _resize_image(image: np.ndarray, size: int) -> np.ndarray:
    if size <= 0:
        return image
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def _convert_color(image: np.ndarray, color_mode: str) -> np.ndarray:
    if color_mode == "rgb":
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if color_mode == "bgr":
        return image
    raise ValueError(f"Unsupported color_mode: {color_mode}")


def _build_lerobot_features(obs: dict[str, Any]) -> dict[str, dict]:
    hw_features: dict[str, Any] = {}
    for key, value in obs.items():
        if isinstance(value, (float, int, np.floating, np.integer)):
            hw_features[key] = float
        elif isinstance(value, np.ndarray):
            hw_features[key] = value.shape
    return hw_to_dataset_features(hw_features, OBS_STR, use_video=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Remote policy inference on a saved frame.")
    parser.add_argument("--frame_dir", required=True, help="Directory with observation.pkl and images")
    parser.add_argument("--server_address", default="127.0.0.1:5555", help="gRPC server address")
    parser.add_argument("--policy_type", default="pi05", help="Policy type")
    parser.add_argument("--policy_path", required=True, help="Policy path on the server")
    parser.add_argument("--policy_device", default="cuda", help="Policy device")
    parser.add_argument("--actions_per_chunk", type=int, default=1, help="Actions per chunk")
    parser.add_argument("--task", default="pick and place", help="Task string")
    parser.add_argument("--rename_map", default="{}", help="JSON rename map")
    parser.add_argument(
        "--output_dir",
        default="",
        help="Local directory to save returned actions (JSON and pickle)",
    )
    parser.add_argument(
        "--timeout_s",
        type=int,
        default=120,
        help="Seconds to wait for actions before timing out",
    )
    parser.add_argument(
        "--color_mode",
        choices=["rgb", "bgr"],
        default="rgb",
        help="Color mode for images (OpenCV capture defaults to RGB in LeRobot)",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=0,
        help="Resize images to NxN before sending (0 keeps original size)",
    )
    args = parser.parse_args()

    obs = _load_obs(args.frame_dir)
    front = _load_image(args.frame_dir, "front")
    wrist = _load_image(args.frame_dir, "wrist")

    if front is not None:
        front = _convert_color(front, args.color_mode)
        if args.resize:
            front = _resize_image(front, args.resize)
        obs["front"] = np.ascontiguousarray(front, dtype=np.uint8)
    if wrist is not None:
        wrist = _convert_color(wrist, args.color_mode)
        if args.resize:
            wrist = _resize_image(wrist, args.resize)
        obs["wrist"] = np.ascontiguousarray(wrist, dtype=np.uint8)

    obs["task"] = args.task
    for key, value in list(obs.items()):
        if isinstance(value, (np.floating, np.integer)):
            obs[key] = float(value)

    lerobot_features = _build_lerobot_features(obs)
    rename_map = json.loads(args.rename_map)

    policy_cfg = RemotePolicyConfig(
        policy_type=args.policy_type,
        pretrained_name_or_path=args.policy_path,
        lerobot_features=lerobot_features,
        actions_per_chunk=args.actions_per_chunk,
        device=args.policy_device,
        rename_map=rename_map,
    )

    channel = grpc.insecure_channel(args.server_address)
    stub = services_pb2_grpc.AsyncInferenceStub(channel)

    print("Connecting to server...")
    stub.Ready(services_pb2.Empty())
    print("✓ Server ready")
    policy_setup = services_pb2.PolicySetup(data=pickle.dumps(policy_cfg))  # nosec
    print("Sending policy instructions...")
    stub.SendPolicyInstructions(policy_setup)
    print("✓ Policy initialized")

    timed_obs = TimedObservation(
        timestamp=time.time(),
        observation=obs,
        timestep=0,
        must_go=True,
    )
    payload = pickle.dumps(timed_obs)  # nosec
    iterator = send_bytes_in_chunks(payload, services_pb2.Observation, silent=True)
    print("Sending observation...")
    stub.SendObservations(iterator)
    print("Waiting for actions...")
    start_wait = time.time()
    try:
        if args.timeout_s <= 0:
            actions_msg = stub.GetActions(services_pb2.Empty())
        else:
            actions_msg = stub.GetActions(
                services_pb2.Empty(),
                timeout=args.timeout_s,
            )
    except grpc.RpcError as exc:
        elapsed = time.time() - start_wait
        print(f"✗ GetActions failed after {elapsed:.1f}s: {exc}")
        print("If this keeps timing out, increase --timeout_s or check server logs.")
        raise
    if len(actions_msg.data) == 0:
        print("No actions returned")
        return 1

    if not torch.cuda.is_available():
        original_load_from_bytes = torch.storage._load_from_bytes

        def _load_from_bytes_cpu(b):
            return torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)

        torch.storage._load_from_bytes = _load_from_bytes_cpu
    try:
        actions = pickle.loads(actions_msg.data)  # nosec
    finally:
        if not torch.cuda.is_available():
            torch.storage._load_from_bytes = original_load_from_bytes
    print(f"Actions returned: {len(actions)}")
    if actions:
        action_tensor = actions[0].get_action()
        print(f"First action shape: {tuple(action_tensor.shape)}")
        print(f"First action values: {action_tensor.tolist()}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = int(time.time() * 1000)
        pkl_path = os.path.join(args.output_dir, f"actions_{timestamp}.pkl")
        json_path = os.path.join(args.output_dir, f"actions_{timestamp}.json")
        with open(pkl_path, "wb") as f:
            pickle.dump(actions, f)  # nosec
        json_payload = [
            {
                "timestep": a.get_timestep(),
                "timestamp": a.get_timestamp(),
                "action": a.get_action().tolist(),
            }
            for a in actions
        ]
        with open(json_path, "w") as f:
            json.dump(json_payload, f, indent=2)
        print(f"Saved actions to {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
