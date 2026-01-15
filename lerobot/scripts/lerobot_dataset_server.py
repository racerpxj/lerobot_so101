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
Serve action chunks from a LeRobot dataset episode over gRPC.

This is a lightweight "dataset server" that reuses the AsyncInference service
message types to stream actions to a replay client.
"""

import logging
import pickle
from concurrent import futures
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

import grpc

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.utils.constants import ACTION
from lerobot.utils.utils import init_logging
try:
    from lerobot.utils.import_utils import register_third_party_plugins
except ImportError:  # pragma: no cover - older lerobot versions
    register_third_party_plugins = None


logger = logging.getLogger(__name__)


@dataclass
class DatasetServerConfig:
    host: str = "127.0.0.1"
    port: int = 5555
    dataset_repo_id: str = "local/lerobot_dataset"
    dataset_root: str | Path | None = None
    episode: int = 0
    actions_per_chunk: int = 10
    fps: int | None = None
    loop: bool = False


class DatasetActionServer(services_pb2_grpc.AsyncInferenceServicer):
    def __init__(self, actions: list[dict[str, float]], fps: int, actions_per_chunk: int, loop: bool) -> None:
        self.actions = actions
        self.fps = fps
        self.actions_per_chunk = max(1, actions_per_chunk)
        self.loop = loop
        self.cursor = 0

    def Ready(self, request, context):  # noqa: N802
        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        for _ in request_iterator:
            pass
        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        if self.cursor >= len(self.actions):
            if not self.loop:
                payload = {"done": True, "actions": [], "fps": self.fps}
                return services_pb2.Actions(data=pickle.dumps(payload))  # nosec
            self.cursor = 0

        start = self.cursor
        end = min(len(self.actions), start + self.actions_per_chunk)
        chunk = self.actions[start:end]
        self.cursor = end
        payload = {"done": False, "actions": chunk, "fps": self.fps}
        return services_pb2.Actions(data=pickle.dumps(payload))  # nosec


def _load_actions(cfg: DatasetServerConfig) -> tuple[list[dict[str, float]], int]:
    dataset = LeRobotDataset(cfg.dataset_repo_id, root=cfg.dataset_root, episodes=[cfg.episode])
    episode_frames = dataset.hf_dataset.filter(lambda x: x["episode_index"] == cfg.episode)
    actions = episode_frames.select_columns(ACTION)
    action_names = dataset.features[ACTION]["names"]

    action_dicts: list[dict[str, float]] = []
    for idx in range(len(episode_frames)):
        action_array = actions[idx][ACTION]
        action_dicts.append({name: float(action_array[i]) for i, name in enumerate(action_names)})

    fps = cfg.fps if cfg.fps is not None else dataset.fps
    return action_dicts, fps


@parser.wrap()
def serve(cfg: DatasetServerConfig) -> None:
    init_logging()
    logging.info(pformat(asdict(cfg)))

    actions, fps = _load_actions(cfg)
    logging.info("Loaded %d actions from episode %d at %d FPS", len(actions), cfg.episode, fps)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    servicer = DatasetActionServer(actions, fps, cfg.actions_per_chunk, cfg.loop)
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(servicer, server)

    server.add_insecure_port(f"{cfg.host}:{cfg.port}")
    server.start()
    logging.info("Dataset server started on %s:%d", cfg.host, cfg.port)
    server.wait_for_termination()


def main() -> None:
    if register_third_party_plugins is not None:
        register_third_party_plugins()
    serve()


if __name__ == "__main__":
    main()
