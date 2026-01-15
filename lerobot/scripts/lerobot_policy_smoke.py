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

import argparse
import json
import pickle  # nosec
import time
from typing import Any

import grpc
import numpy as np

from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import send_bytes_in_chunks
from lerobot.utils.constants import OBS_STR


def _parse_shape(value: str) -> tuple[int, int, int]:
    parts = value.split("x")
    if len(parts) != 3:
        raise ValueError(f"Invalid shape '{value}'. Use HxWxC, e.g. 224x224x3.")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def _build_lerobot_features(front_shape: tuple[int, int, int], wrist_shape: tuple[int, int, int]):
    joints = [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ]
    hw_features: dict[str, Any] = {name: float for name in joints}
    hw_features["front"] = front_shape
    hw_features["wrist"] = wrist_shape
    return hw_to_dataset_features(hw_features, OBS_STR, use_video=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Remote policy inference smoke test (no robot).")
    parser.add_argument("--address", default="127.0.0.1:5555", help="gRPC server address")
    parser.add_argument("--policy-path", required=True, help="Remote policy path on server")
    parser.add_argument("--policy-type", default="pi05", help="Policy type, e.g. pi05")
    parser.add_argument("--policy-device", default="cuda", help="Policy device, e.g. cuda or cpu")
    parser.add_argument("--actions-per-chunk", type=int, default=1, help="Actions per chunk")
    parser.add_argument("--front-shape", default="224x224x3", help="Front camera shape HxWxC")
    parser.add_argument("--wrist-shape", default="224x224x3", help="Wrist camera shape HxWxC")
    parser.add_argument("--task", default="dummy task", help="Task string")
    parser.add_argument("--rename-map", default="{}", help='JSON map for renaming observation keys')
    args = parser.parse_args()

    rename_map = json.loads(args.rename_map)
    front_shape = _parse_shape(args.front_shape)
    wrist_shape = _parse_shape(args.wrist_shape)
    lerobot_features = _build_lerobot_features(front_shape, wrist_shape)

    policy_cfg = RemotePolicyConfig(
        policy_type=args.policy_type,
        pretrained_name_or_path=args.policy_path,
        lerobot_features=lerobot_features,
        actions_per_chunk=args.actions_per_chunk,
        device=args.policy_device,
        rename_map=rename_map,
    )

    channel = grpc.insecure_channel(args.address)
    stub = services_pb2_grpc.AsyncInferenceStub(channel)

    stub.Ready(services_pb2.Empty())
    policy_setup = services_pb2.PolicySetup(data=pickle.dumps(policy_cfg))  # nosec
    stub.SendPolicyInstructions(policy_setup)

    obs = {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": 0.0,
        "elbow_flex.pos": 0.0,
        "wrist_flex.pos": 0.0,
        "wrist_roll.pos": 0.0,
        "gripper.pos": 0.0,
        "front": np.zeros(front_shape, dtype=np.uint8),
        "wrist": np.zeros(wrist_shape, dtype=np.uint8),
        "task": args.task,
    }

    timed_obs = TimedObservation(
        timestamp=time.time(),
        observation=obs,
        timestep=0,
        must_go=True,
    )
    payload = pickle.dumps(timed_obs)  # nosec
    iterator = send_bytes_in_chunks(payload, services_pb2.Observation, silent=True)
    stub.SendObservations(iterator)

    actions_msg = stub.GetActions(services_pb2.Empty())
    if len(actions_msg.data) == 0:
        print("No actions returned")
        return 1

    actions = pickle.loads(actions_msg.data)  # nosec
    print(f"Actions returned: {len(actions)}")
    if actions:
        action_tensor = actions[0].get_action()
        print(f"First action shape: {tuple(action_tensor.shape)}")
        print(f"First action values: {action_tensor.tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
