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
Replay actions from a remote dataset server on a real robot with verbose logging.
"""

import logging
import pickle
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import grpc

from lerobot.configs import parser
from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
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
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging


logger = logging.getLogger(__name__)


@dataclass
class ReplayDebugClientConfig:
    robot: RobotConfig
    server_address: str = "127.0.0.1:5555"
    fps: int | None = None
    timeout_s: float = 30.0
    log_every: int = 1


def _load_actions_chunk(stub: services_pb2_grpc.AsyncInferenceStub, timeout_s: float | None):
    return stub.GetActions(services_pb2.Empty(), timeout=timeout_s)


@parser.wrap()
def replay(cfg: ReplayDebugClientConfig) -> None:
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot_action_processor = make_default_robot_action_processor()
    robot = make_robot_from_config(cfg.robot)
    logging.info("Calibration file path: %s", robot.calibration_fpath)

    channel = grpc.insecure_channel(cfg.server_address)
    stub = services_pb2_grpc.AsyncInferenceStub(channel)

    stub.Ready(services_pb2.Empty())
    robot.connect()

    step_index = 0
    try:
        while True:
            actions_msg = _load_actions_chunk(stub, cfg.timeout_s if cfg.timeout_s > 0 else None)
            if not actions_msg.data:
                continue
            payload = pickle.loads(actions_msg.data)  # nosec
            done = payload.get("done", False)
            actions = payload.get("actions", [])
            fps = payload.get("fps") or cfg.fps
            if fps is None:
                fps = 30

            if done and not actions:
                logging.info("Replay finished.")
                break

            for action in actions:
                step_index += 1
                if cfg.log_every > 0 and step_index % cfg.log_every == 0:
                    logging.info("Action %d: %s", step_index, action)

                step_start = time.perf_counter()
                robot_obs = robot.get_observation()
                processed_action = robot_action_processor((action, robot_obs))
                _ = robot.send_action(processed_action)
                dt_s = time.perf_counter() - step_start
                precise_sleep(1 / fps - dt_s)
    finally:
        robot.disconnect()


def main() -> None:
    register_third_party_plugins()
    replay()


if __name__ == "__main__":
    main()
