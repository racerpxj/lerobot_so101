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
Serve policy-inferred actions from dataset states + videos over gRPC.

Remote workflow:
  dataset -> PolicyServer inference -> action queue -> client replay (robot)
"""

import io
import logging
import pickle  # nosec
import threading
import time
from concurrent import futures
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from queue import Empty, Queue
from typing import Any

import grpc
import numpy as np
import torch

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.utils.utils import init_logging
from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation

try:
    from lerobot.utils.import_utils import register_third_party_plugins
except ImportError:  # pragma: no cover - older lerobot versions
    register_third_party_plugins = None


logger = logging.getLogger(__name__)


@dataclass
class DatasetPolicyInferServerConfig:
    host: str = "127.0.0.1"
    port: int = 5556
    # Dataset
    dataset_repo_id: str = "local/lerobot_dataset"
    dataset_root: str | None = None
    episode: int = 0
    image_source: str = "dataset"  # zeros|dataset
    video_backend: str | None = None
    task: str | None = None
    max_steps: int | None = None
    loop: bool = False
    # Policy server
    policy_server_address: str = "127.0.0.1:5555"
    policy_type: str = "pi05"
    pretrained_name_or_path: str = ""
    policy_device: str = "cuda"
    actions_per_chunk: int = 1
    timeout_s: float = 30.0
    # Output control
    output_fps: int | None = None
    debug_shapes: bool = False
    preview_dir: str | None = None
    preview_every: int = 1
    preview_overwrite: bool = True


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
            img = dataset_row[image_key]
            if torch.is_tensor(img):
                img = img.cpu().numpy()
            img = np.asarray(img)
            # Normalize to HWC so async_inference helpers don't interpret width as channels.
            if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
                img = np.transpose(img, (1, 2, 0))
            raw_obs[cam_name] = img
        else:
            h, w, c = _image_shape(features, image_key)
            raw_obs[cam_name] = np.zeros((h, w, c), dtype=np.uint8)

    if task:
        raw_obs["task"] = task

    return raw_obs


def _action_tensor_to_dict(action_tensor: torch.Tensor, action_names: list[str]) -> dict[str, float]:
    return {name: float(action_tensor[i].item()) for i, name in enumerate(action_names)}


def _save_preview_images(preview_dir: str, index: int, images: dict[str, Any], overwrite: bool) -> None:
    try:
        import cv2  # type: ignore
    except Exception:
        logger.warning("OpenCV not available; preview images will not be saved.")
        return

    Path(preview_dir).mkdir(parents=True, exist_ok=True)
    for name, img in images.items():
        if img is None:
            continue
        arr = np.asarray(img)
        if arr.ndim != 3:
            continue
        # Normalize dtype for imwrite
        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            if arr.max() <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        # Convert RGB -> BGR for cv2.imwrite.
        if arr.shape[-1] == 3:
            arr = arr[..., ::-1]
        if overwrite:
            out_path = Path(preview_dir) / f"latest_{name}.jpg"
        else:
            out_path = Path(preview_dir) / f"{name}_{index:06d}.jpg"
        cv2.imwrite(str(out_path), arr)


class DatasetPolicyInferServer(services_pb2_grpc.AsyncInferenceServicer):
    def __init__(self, cfg: DatasetPolicyInferServerConfig):
        self.cfg = cfg
        self.shutdown_event = threading.Event()
        self.action_queue: Queue[dict[str, Any]] = Queue()
        self.worker_thread = threading.Thread(target=self._run_inference_loop, daemon=True)
        self.worker_thread.start()

    def Ready(self, request, context):  # noqa: N802
        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        for _ in request_iterator:
            pass
        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        try:
            payload = self.action_queue.get(timeout=5.0)
        except Empty:
            return services_pb2.Actions(data=b"")

        return services_pb2.Actions(data=pickle.dumps(payload))  # nosec

    def _run_inference_loop(self) -> None:
        dataset = LeRobotDataset(
            self.cfg.dataset_repo_id,
            root=self.cfg.dataset_root,
            episodes=[self.cfg.episode],
            video_backend=self.cfg.video_backend,
        )
        features = dataset.meta.features
        state_names = features[OBS_STATE]["names"]
        image_keys = _dataset_image_keys(features)
        action_names = features[ACTION]["names"]

        channel = grpc.insecure_channel(self.cfg.policy_server_address, grpc_channel_options())
        stub = services_pb2_grpc.AsyncInferenceStub(channel)
        self._wait_for_policy_server(stub)

        policy_config = RemotePolicyConfig(
            self.cfg.policy_type,
            self.cfg.pretrained_name_or_path,
            features,
            self.cfg.actions_per_chunk,
            self.cfg.policy_device,
        )
        policy_setup = services_pb2.PolicySetup(data=pickle.dumps(policy_config))
        stub.SendPolicyInstructions(policy_setup)

        output_fps = self.cfg.output_fps if self.cfg.output_fps is not None else dataset.fps

        while not self.shutdown_event.is_set():
            total_frames = len(dataset)
            if self.cfg.max_steps is not None:
                total_frames = min(total_frames, self.cfg.max_steps)

            for idx in range(total_frames):
                if self.shutdown_event.is_set():
                    break

                row = dataset[idx]
                state_values = np.asarray(row[OBS_STATE], dtype=np.float32)
                task_text = self.cfg.task if self.cfg.task is not None else row.get("task")
                raw_obs = _build_raw_observation(
                    state_names=state_names,
                    state_values=state_values,
                    image_keys=image_keys,
                    features=features,
                    image_source=self.cfg.image_source,
                    dataset_row=row,
                    task=task_text,
                )
                if self.cfg.debug_shapes and idx == 0:
                    for cam_key in image_keys:
                        name = cam_key.removeprefix("observation.images.")
                        arr = raw_obs.get(name)
                        if arr is not None:
                            logger.info("Image %s shape: %s", name, getattr(arr, "shape", None))
                if self.cfg.preview_dir and (idx % max(1, self.cfg.preview_every) == 0):
                    preview_images = {
                        cam_key.removeprefix("observation.images."): raw_obs.get(
                            cam_key.removeprefix("observation.images.")
                        )
                        for cam_key in image_keys
                    }
                    _save_preview_images(self.cfg.preview_dir, idx, preview_images, self.cfg.preview_overwrite)

                timed_obs = TimedObservation(
                    timestamp=time.time(),
                    timestep=idx,
                    observation=raw_obs,
                    must_go=True,
                )

                obs_bytes = pickle.dumps(timed_obs)
                obs_iter = send_bytes_in_chunks(
                    obs_bytes, services_pb2.Observation, log_prefix="[SERVER] Observation", silent=True
                )
                _ = stub.SendObservations(obs_iter)

                actions_msg = stub.GetActions(
                    services_pb2.Empty(), timeout=self.cfg.timeout_s if self.cfg.timeout_s > 0 else None
                )
                if len(actions_msg.data) == 0:
                    continue

                timed_actions = pickle.loads(actions_msg.data)  # nosec
                actions = [_action_tensor_to_dict(a.get_action(), action_names) for a in timed_actions]
                payload = {"done": False, "actions": actions, "fps": output_fps}
                self.action_queue.put(payload)

            self.action_queue.put({"done": True, "actions": [], "fps": output_fps})
            if not self.cfg.loop:
                break

    def _wait_for_policy_server(self, stub: services_pb2_grpc.AsyncInferenceStub) -> None:
        for attempt in range(1, 31):
            try:
                stub.Ready(services_pb2.Empty(), timeout=3.0)
                logger.info("Connected to policy server at %s", self.cfg.policy_server_address)
                return
            except grpc.RpcError as exc:
                logger.warning(
                    "Policy server not ready (%d/30): %s", attempt, getattr(exc, "details", lambda: exc)()
                )
                time.sleep(1.0)
        raise RuntimeError(f"Policy server not reachable at {self.cfg.policy_server_address}")


@parser.wrap()
def serve(cfg: DatasetPolicyInferServerConfig) -> None:
    init_logging()
    logging.info(pformat(asdict(cfg)))

    if not cfg.pretrained_name_or_path:
        raise ValueError("pretrained_name_or_path must be provided")

    if cfg.image_source not in {"zeros", "dataset"}:
        raise ValueError("image_source must be one of: zeros, dataset")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(DatasetPolicyInferServer(cfg), server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")
    server.start()
    logging.info("Dataset policy infer server started on %s:%d", cfg.host, cfg.port)
    server.wait_for_termination()


def main() -> None:
    if register_third_party_plugins is not None:
        register_third_party_plugins()
    serve()


if __name__ == "__main__":
    main()
