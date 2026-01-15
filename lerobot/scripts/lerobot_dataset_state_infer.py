# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
Use dataset states + videos as observations to run remote policy inference and drive a real robot.

This script sends dataset-derived observations to a PolicyServer, receives action chunks,
and executes actions on the robot. It does not use live cameras.
"""

import io
import logging
import pickle  # nosec
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

import grpc
import numpy as np
import torch

from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import make_default_robot_action_processor
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    bi_so100_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so100_follower,
    so101_follower,
)
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation

try:
    from lerobot.utils.import_utils import register_third_party_plugins
except ImportError:  # pragma: no cover - older lerobot versions
    register_third_party_plugins = None


logger = logging.getLogger(__name__)


@dataclass
class DatasetStateInferConfig:
    # Dataset
    dataset_repo_id: str
    dataset_root: str | None = None
    episode: int = 0
    image_source: str = "zeros"  # zeros|dataset
    task: str | None = None
    max_steps: int | None = None

    # Policy server
    server_address: str = "127.0.0.1:5555"
    policy_type: str = "pi05"
    pretrained_name_or_path: str = ""
    policy_device: str = "cuda"
    actions_per_chunk: int = 1

    # Robot
    robot: RobotConfig | None = None
    control_fps: int | None = None
    disable_cameras: bool = True

    # Client
    timeout_s: float = 30.0
    force_cpu_actions: bool = True


def _patch_torch_load_to_cpu() -> None:
    original = torch.storage._load_from_bytes

    def _load_from_bytes_cpu(b):
        return torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)

    torch.storage._load_from_bytes = _load_from_bytes_cpu
    return original


def _dataset_image_keys(features: dict[str, dict]) -> list[str]:
    return [k for k in features if k.startswith(OBS_IMAGES)]


def _image_shape(features: dict[str, dict], image_key: str) -> tuple[int, int, int]:
    shape = features[image_key]["shape"]
    return (int(shape[0]), int(shape[1]), int(shape[2]))


def _build_raw_observation(
    state_names: list[str],
    state_values: np.ndarray,
    image_keys: list[str],
    features: dict[str, dict],
    image_source: str,
    dataset_row: dict[str, Any] | None,
    task: str | None,
) -> dict[str, Any]:
    raw_obs: dict[str, Any] = {name: float(state_values[i]) for i, name in enumerate(state_names)}

    for image_key in image_keys:
        cam_name = image_key.removeprefix("observation.images.")
        if image_source == "dataset" and dataset_row is not None:
            raw_obs[cam_name] = dataset_row[image_key]
        else:
            h, w, c = _image_shape(features, image_key)
            raw_obs[cam_name] = np.zeros((h, w, c), dtype=np.uint8)

    if task:
        raw_obs["task"] = task

    return raw_obs


@parser.wrap()
def run(cfg: DatasetStateInferConfig) -> None:
    init_logging()
    logging.info(pformat(asdict(cfg)))

    if cfg.robot is None:
        raise ValueError("robot config must be provided")
    if not cfg.pretrained_name_or_path:
        raise ValueError("pretrained_name_or_path must be provided")

    if cfg.dataset_root is not None:
        root_path = Path(cfg.dataset_root)
        if not root_path.exists():
            raise RuntimeError(
                f"dataset_root not found: {cfg.dataset_root}. "
                "This path must exist on the machine running this script. "
                "Mount or copy the dataset locally, or run this script on the server that has the dataset."
            )
        if not root_path.is_dir():
            raise RuntimeError(f"dataset_root is not a directory: {cfg.dataset_root}")

    dataset = LeRobotDataset(cfg.dataset_repo_id, root=cfg.dataset_root, episodes=[cfg.episode])
    features = dataset.meta.features

    state_key = OBS_STATE
    state_names = features[state_key]["names"]
    image_keys = _dataset_image_keys(features)

    if cfg.image_source not in {"zeros", "dataset"}:
        raise ValueError("image_source must be one of: zeros, dataset")

    if cfg.disable_cameras and hasattr(cfg.robot, "cameras"):
        cfg.robot.cameras = {}

    robot_action_processor = make_default_robot_action_processor()
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    logging.info("Calibration file path: %s", robot.calibration_fpath)

    channel = grpc.insecure_channel(cfg.server_address, grpc_channel_options())
    stub = services_pb2_grpc.AsyncInferenceStub(channel)

    stub.Ready(services_pb2.Empty())
    policy_config = RemotePolicyConfig(
        cfg.policy_type,
        cfg.pretrained_name_or_path,
        features,
        cfg.actions_per_chunk,
        cfg.policy_device,
    )
    policy_setup = services_pb2.PolicySetup(data=pickle.dumps(policy_config))
    stub.SendPolicyInstructions(policy_setup)

    if cfg.force_cpu_actions and not torch.cuda.is_available():
        original_loader = _patch_torch_load_to_cpu()
    else:
        original_loader = None

    control_fps = cfg.control_fps if cfg.control_fps is not None else dataset.fps
    total_frames = len(dataset)
    if cfg.max_steps is not None:
        total_frames = min(total_frames, cfg.max_steps)

    try:
        for idx in range(total_frames):
            row = dataset[idx]
            state_values = np.asarray(row[state_key], dtype=np.float32)
            task_text = cfg.task if cfg.task is not None else row.get("task")
            raw_obs = _build_raw_observation(
                state_names=state_names,
                state_values=state_values,
                image_keys=image_keys,
                features=features,
                image_source=cfg.image_source,
                dataset_row=row,
                task=task_text,
            )

            timed_obs = TimedObservation(
                timestamp=time.time(),
                timestep=idx,
                observation=raw_obs,
                must_go=True,
            )

            obs_bytes = pickle.dumps(timed_obs)
            obs_iter = send_bytes_in_chunks(
                obs_bytes, services_pb2.Observation, log_prefix="[CLIENT] Observation", silent=True
            )
            _ = stub.SendObservations(obs_iter)

            actions_msg = stub.GetActions(services_pb2.Empty(), timeout=cfg.timeout_s if cfg.timeout_s > 0 else None)
            if len(actions_msg.data) == 0:
                continue

            timed_actions = pickle.loads(actions_msg.data)  # nosec
            for timed_action in timed_actions:
                action_tensor = timed_action.get_action()
                action = {key: action_tensor[i].item() for i, key in enumerate(robot.action_features)}
                processed_action = robot_action_processor((action, None))
                _ = robot.send_action(processed_action)
                precise_sleep(1 / control_fps)
    finally:
        if original_loader is not None:
            torch.storage._load_from_bytes = original_loader
        robot.disconnect()


def main() -> None:
    if register_third_party_plugins is not None:
        register_third_party_plugins()
    run()


if __name__ == "__main__":
    main()
