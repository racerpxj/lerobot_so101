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
Record a dataset locally while running policy inference remotely.

Example (local machine):
```shell
python lerobot_record_client.py \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5AB01587521 \
  --robot.cameras='{"front": {"type": "opencv", "index_or_path": 1}}' \
  --dataset.repo_id=username/dataset_name \
  --dataset.single_task="Pick and place" \
  --dataset.num_episodes=50 \
  --remote_policy.server_address=127.0.0.1:8080 \
  --remote_policy.policy_type=pi05 \
  --remote_policy.pretrained_name_or_path=/remote/path/to/pi05 \
  --remote_policy.policy_device=cuda
```
"""

import logging
import io
import pickle  # nosec
import sys
import time
import types
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import grpc
import torch


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

from lerobot.async_inference.helpers import (
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    map_robot_keys_to_lerobot_features,
)
from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import (  # noqa: F401
    Robot,
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
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_so100_leader,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    so100_leader,
    so101_leader,
)
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)

try:
    from lerobot.utils.import_utils import register_third_party_plugins
except Exception:
    def register_third_party_plugins() -> None:
        return None

from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


@dataclass
class DatasetRecordConfig:
    """Configuration for dataset recording."""
    repo_id: str
    single_task: str
    root: str | Path | None = None
    fps: int = 30
    episode_time_s: int | float = 60
    reset_time_s: int | float = 60
    num_episodes: int = 50
    video: bool = True
    push_to_hub: bool = True
    private: bool = False
    tags: list[str] | None = None
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4
    video_encoding_batch_size: int = 1
    rename_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.single_task is None:
            raise ValueError("You need to provide a task as argument in `single_task`.")


@dataclass
class RemotePolicyClientConfig:
    """Configuration for remote policy client."""
    server_address: str
    policy_type: str
    pretrained_name_or_path: str
    policy_device: str = "cpu"
    actions_per_chunk: int = 1
    connection_timeout: float = 30.0
    retry_attempts: int = 3
    image_resize: int = 0

    def __post_init__(self):
        if not self.server_address:
            raise ValueError("remote_policy.server_address is required.")
        if not self.policy_type:
            raise ValueError("remote_policy.policy_type is required.")
        if not self.pretrained_name_or_path:
            raise ValueError("remote_policy.pretrained_name_or_path is required.")
        if self.actions_per_chunk <= 0:
            raise ValueError("remote_policy.actions_per_chunk must be >= 1.")
        if self.image_resize < 0:
            raise ValueError("remote_policy.image_resize must be >= 0.")


@dataclass
class RecordConfig:
    """Main configuration for recording."""
    robot: RobotConfig
    dataset: DatasetRecordConfig
    teleop: TeleoperatorConfig | None = None
    remote_policy: RemotePolicyClientConfig | None = None
    display_data: bool = False
    play_sounds: bool = True
    resume: bool = False

    def __post_init__(self):
        if self.teleop is None and self.remote_policy is None:
            raise ValueError("Choose a remote_policy, a teleoperator or both to control the robot")


class RemotePolicyClient:
    """Client for remote policy inference."""
    
    def __init__(
        self,
        config: RemotePolicyClientConfig,
        lerobot_features: dict[str, Any],
        rename_map: dict[str, str],
        fps: int,
    ):
        self.config = config
        self.policy_config = RemotePolicyConfig(
            policy_type=config.policy_type,
            pretrained_name_or_path=config.pretrained_name_or_path,
            lerobot_features=lerobot_features,
            actions_per_chunk=config.actions_per_chunk,
            device=config.policy_device,
            rename_map=rename_map,
        )
        
        # Create gRPC channel with appropriate options
        self.channel = grpc.insecure_channel(
            config.server_address, 
            grpc_channel_options(initial_backoff=f"{1 / fps:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self._timestep = 0
        self._is_connected = False
        self._action_queue: deque[TimedAction] = deque()
        
        logging.info(f"Remote policy client initialized for {config.server_address}")

    def connect(self) -> None:
        """Connect to the remote policy server and initialize policy."""
        logging.info("")
        logging.info("=" * 70)
        logging.info("Connecting to Remote Policy Server")
        logging.info("=" * 70)
        logging.info(f"Server address: {self.config.server_address}")
        logging.info(f"Policy type: {self.config.policy_type}")
        logging.info(f"Pretrained path: {self.config.pretrained_name_or_path}")
        logging.info(f"Device: {self.config.policy_device}")
        logging.info("=" * 70)
        
        # Check server readiness
        for attempt in range(self.config.retry_attempts):
            try:
                logging.info(f"Attempt {attempt + 1}/{self.config.retry_attempts}: Checking server readiness...")
                self.stub.Ready(services_pb2.Empty())
                logging.info("✓ Server is ready")
                break
            except grpc.RpcError as e:
                error_msg = e.details() if hasattr(e, 'details') else str(e)
                logging.error(f"✗ Connection attempt {attempt + 1} failed: {error_msg}")
                if attempt < self.config.retry_attempts - 1:
                    wait_time = 2 ** attempt
                    logging.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(
                        f"Failed to connect to server at {self.config.server_address} "
                        f"after {self.config.retry_attempts} attempts.\n\n"
                        f"Please ensure:\n"
                        f"  1. Server is running: python lerobot_record_server.py --host=0.0.0.0 --port=8080\n"
                        f"  2. SSH tunnel is active (if using remote server): ssh -L 8080:127.0.0.1:8080 user@host\n"
                        f"  3. Server address is correct: {self.config.server_address}\n"
                        f"  4. No firewall blocking the connection"
                    ) from e
        
        # Send policy configuration
        try:
            logging.info("Sending policy configuration to server...")
            policy_setup = services_pb2.PolicySetup(data=pickle.dumps(self.policy_config))  # nosec
            self.stub.SendPolicyInstructions(policy_setup)
            self._is_connected = True
            logging.info("✓ Policy initialized on remote server successfully")
            logging.info("=" * 70)
            logging.info("")
        except grpc.RpcError as e:
            error_details = e.details() if hasattr(e, 'details') else str(e)
            logging.error("")
            logging.error("=" * 70)
            logging.error("✗ Failed to initialize policy on server")
            logging.error("=" * 70)
            logging.error(f"Error: {error_details}")
            logging.error("=" * 70)
            
            # Provide helpful error messages based on common issues
            if "SiglipVisionConfig" in error_details:
                raise RuntimeError(
                    f"Policy initialization failed due to missing Siglip modules on server.\n"
                    f"Server error: {error_details}\n\n"
                    f"Fix on server:\n"
                    f"  pip install transformers>=4.40.0 --upgrade\n"
                    f"  pip install torch torchvision --upgrade"
                ) from e
            elif "pretrained" in error_details.lower() or "does not exist" in error_details.lower():
                raise RuntimeError(
                    f"Policy initialization failed - model path may be incorrect.\n"
                    f"Server error: {error_details}\n\n"
                    f"Please verify on SERVER machine:\n"
                    f"  1. Model path exists: {self.config.pretrained_name_or_path}\n"
                    f"  2. Model files are accessible (config.json, *.safetensors)\n"
                    f"  3. Model is compatible with policy type: {self.config.policy_type}"
                ) from e
            else:
                raise RuntimeError(
                    f"Failed to initialize policy on server.\n"
                    f"Server error: {error_details}\n\n"
                    f"Check server logs for more details."
                ) from e

    def close(self) -> None:
        """Close the connection to the remote policy server."""
        if self._is_connected:
            logging.info("Closing connection to remote policy server...")
            self.channel.close()
            self._is_connected = False
        self._action_queue.clear()

    def reset(self) -> None:
        """Reset local client state between episodes."""
        self._timestep = 0
        self._action_queue.clear()

    def _send_observation(self, observation: TimedObservation) -> None:
        """Send observation to the remote server."""
        payload = pickle.dumps(observation)  # nosec
        iterator = send_bytes_in_chunks(
            payload, 
            services_pb2.Observation, 
            log_prefix="[CLIENT] Observation", 
            silent=True
        )
        self.stub.SendObservations(iterator)

    def _maybe_resize_images(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self.config.image_resize <= 0:
            return observation
        resized = dict(observation)
        size = self.config.image_resize
        for key, value in observation.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            if isinstance(value, (list, tuple)):
                continue
            if hasattr(value, "ndim") and getattr(value, "ndim", 0) == 3:
                try:
                    resized[key] = cv2.resize(value, (size, size), interpolation=cv2.INTER_AREA)
                except Exception:
                    resized[key] = value
        return resized

    def _receive_actions(self) -> list[TimedAction]:
        """Receive actions from the remote server."""
        try:
            actions_msg = self.stub.GetActions(services_pb2.Empty())
            if len(actions_msg.data) == 0:
                return []
            if not torch.cuda.is_available():
                original_load_from_bytes = torch.storage._load_from_bytes

                def _load_from_bytes_cpu(b):
                    return torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)

                torch.storage._load_from_bytes = _load_from_bytes_cpu
            try:
                return pickle.loads(actions_msg.data)  # nosec
            finally:
                if not torch.cuda.is_available():
                    torch.storage._load_from_bytes = original_load_from_bytes
        except grpc.RpcError as e:
            logging.error(f"Failed to receive actions: {e}")
            return []

    def predict_action(self, observation: dict[str, Any], task: str | None) -> torch.Tensor | None:
        """
        Send observation and get predicted action from remote policy.
        
        Args:
            observation: Robot observation dictionary
            task: Task description
            
        Returns:
            Action tensor or None if no action available
        """
        if self._action_queue:
            timed_action = self._action_queue.popleft()
            self._timestep = max(self._timestep, timed_action.get_timestep() + 1)
            return timed_action.get_action()

        obs_payload = self._maybe_resize_images(observation)
        if task is not None:
            obs_payload["task"] = task

        timed_obs = TimedObservation(
            timestamp=time.time(),
            observation=obs_payload,
            timestep=self._timestep,
            must_go=True,
        )

        self._send_observation(timed_obs)
        actions = self._receive_actions()
        if not actions:
            return None

        self._action_queue.extend(actions)
        timed_action = self._action_queue.popleft()
        self._timestep = max(self._timestep, timed_action.get_timestep() + 1)
        return timed_action.get_action()


def _action_tensor_to_action_dict(
    action_tensor: torch.Tensor, 
    action_names: list[str]
) -> dict[str, float]:
    """Convert action tensor to dictionary format."""
    if action_tensor.ndim > 1:
        action_tensor = action_tensor.squeeze(0)
    action_tensor = action_tensor.to("cpu")
    return {name: float(action_tensor[i]) for i, name in enumerate(action_names)}


@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | list[Teleoperator] | None = None,
    remote_policy: RemotePolicyClient | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
):
    """Main recording loop."""
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    if remote_policy is not None:
        remote_policy.reset()

    # Handle multi-teleop setup
    teleop_arm = None
    teleop_keyboard = None
    if isinstance(teleop, list):
        teleop_keyboard = next((t for t in teleop if isinstance(t, KeyboardTeleop)), None)
        teleop_arm = next(
            (
                t for t in teleop
                if isinstance(
                    t,
                    (
                        so100_leader.SO100Leader,
                        so101_leader.SO101Leader,
                        koch_leader.KochLeader,
                        omx_leader.OmxLeader,
                    ),
                )
            ),
            None,
        )

        if not (teleop_arm and teleop_keyboard and len(teleop) == 2 and robot.name == "lekiwi_client"):
            raise ValueError(
                "For multi-teleop, the list must contain exactly one KeyboardTeleop and one arm teleoperator. "
                "Currently only supported for LeKiwi robot."
            )

    timestamp = 0
    start_episode_t = time.perf_counter()
    
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        # Get robot observation
        obs = robot.get_observation()

        # Process observation
        obs_processed = robot_observation_processor(obs)

        if remote_policy is not None or dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        # Get action from remote policy or teleop
        act_processed_policy = None
        act_processed_teleop = None
        
        if remote_policy is not None:
            action_tensor = remote_policy.predict_action(obs, task=single_task)
            if action_tensor is None:
                # No action available yet, skip this iteration
                precise_sleep(1 / fps)
                continue
            
            action_names = dataset.features[ACTION]["names"]
            act_processed_policy = _action_tensor_to_action_dict(action_tensor, action_names)
            
        elif isinstance(teleop, Teleoperator):
            act = teleop.get_action()
            act_processed_teleop = teleop_action_processor((act, obs))
            
        elif isinstance(teleop, list):
            # Handle multi-teleop (arm + keyboard)
            arm_action = teleop_arm.get_action()
            arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
            keyboard_action = teleop_keyboard.get_action()
            base_action = robot._from_keyboard_to_base_action(keyboard_action)
            act = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
            act_processed_teleop = teleop_action_processor((act, obs))
        else:
            logging.info(
                "No remote policy or teleoperator provided, skipping action generation. "
                "This is likely during environment reset without a teleop device."
            )
            continue

        # Determine which action to use
        if act_processed_policy is not None:
            action_values = act_processed_policy
            robot_action_to_send = robot_action_processor((act_processed_policy, obs))
        else:
            action_values = act_processed_teleop
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))

        # Send action to robot
        _sent_action = robot.send_action(robot_action_to_send)

        # Write to dataset
        if dataset is not None:
            action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)

        # Display data if requested
        if display_data:
            log_rerun_data(observation=obs_processed, action=action_values)

        # Maintain loop timing
        dt_s = time.perf_counter() - start_loop_t
        precise_sleep(1 / fps - dt_s)

        timestamp = time.perf_counter() - start_episode_t


def record(
    robot: Robot,
    dataset_config: DatasetRecordConfig,
    remote_policy_config: RemotePolicyClientConfig | None = None,
    teleop: Teleoperator | None = None,
    display_data: bool = False,
    play_sounds: bool = True,
    resume: bool = False,
) -> LeRobotDataset:
    """Main recording function."""
    init_logging()
    logging.info("")
    logging.info("#" * 80)
    logging.info("# LeRobot Remote Recording Client")
    logging.info("#" * 80)
    logging.info("")
    
    # Validate configuration
    if teleop is None and remote_policy_config is None:
        raise ValueError("Either teleop or remote_policy must be provided")
    
    logging.info(f"Robot type: {robot.name}")
    logging.info(f"Dataset: {dataset_config.repo_id}")
    logging.info(f"Task: {dataset_config.single_task}")
    if remote_policy_config:
        logging.info(f"Remote policy: {remote_policy_config.policy_type} @ {remote_policy_config.server_address}")
    logging.info("")
    
    if display_data:
        init_rerun(session_name="recording")

    # Initialize processors
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # Aggregate features from processors
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=dataset_config.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=dataset_config.video,
        ),
    )

    dataset = None
    listener = None
    remote_policy = None

    try:
        # Create or resume dataset
        if resume:
            logging.info(f"Resuming dataset: {dataset_config.repo_id}")
            dataset = LeRobotDataset(
                dataset_config.repo_id,
                root=dataset_config.root,
                batch_encoding_size=dataset_config.video_encoding_batch_size,
            )

            if hasattr(robot, "cameras") and len(robot.cameras) > 0:
                dataset.start_image_writer(
                    num_processes=dataset_config.num_image_writer_processes,
                    num_threads=dataset_config.num_image_writer_threads_per_camera * len(robot.cameras),
                )
            sanity_check_dataset_robot_compatibility(dataset, robot, dataset_config.fps, dataset_features)
        else:
            # Create new dataset
            policy_cfg_for_check = None
            if remote_policy_config is not None:
                policy_cfg_for_check = SimpleNamespace(type=remote_policy_config.policy_type)
            sanity_check_dataset_name(dataset_config.repo_id, policy_cfg_for_check)
            
            logging.info(f"Creating new dataset: {dataset_config.repo_id}")
            dataset = LeRobotDataset.create(
                dataset_config.repo_id,
                dataset_config.fps,
                root=dataset_config.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=dataset_config.video,
                image_writer_processes=dataset_config.num_image_writer_processes,
                image_writer_threads=dataset_config.num_image_writer_threads_per_camera * len(robot.cameras),
                batch_encoding_size=dataset_config.video_encoding_batch_size,
            )

        # Initialize remote policy client
        if remote_policy_config is not None:
            logging.info("Initializing remote policy client...")
            lerobot_features = map_robot_keys_to_lerobot_features(robot)
            remote_policy = RemotePolicyClient(
                config=remote_policy_config,
                lerobot_features=lerobot_features,
                rename_map=dataset_config.rename_map,
                fps=dataset_config.fps,
            )

        # Connect all components
        logging.info("Connecting to robot...")
        robot.connect()
        
        if teleop is not None:
            logging.info("Connecting to teleoperator...")
            teleop.connect()
            
        if remote_policy is not None:
            remote_policy.connect()

        # Initialize keyboard listener
        listener, events = init_keyboard_listener()

        # Main recording loop
        logging.info("")
        logging.info("=" * 80)
        logging.info("Starting Recording")
        logging.info("=" * 80)
        logging.info("")
        
        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < dataset_config.num_episodes and not events["stop_recording"]:
                log_say(f"Recording episode {dataset.num_episodes}", play_sounds)
                
                record_loop(
                    robot=robot,
                    events=events,
                    fps=dataset_config.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    remote_policy=remote_policy,
                    dataset=dataset,
                    control_time_s=dataset_config.episode_time_s,
                    single_task=dataset_config.single_task,
                    display_data=display_data,
                )

                # Reset environment
                if not events["stop_recording"] and (
                    (recorded_episodes < dataset_config.num_episodes - 1) or events["rerecord_episode"]
                ):
                    log_say("Reset the environment", play_sounds)
                    record_loop(
                        robot=robot,
                        events=events,
                        fps=dataset_config.fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=teleop,
                        control_time_s=dataset_config.reset_time_s,
                        single_task=dataset_config.single_task,
                        display_data=display_data,
                    )

                # Handle re-recording
                if events["rerecord_episode"]:
                    log_say("Re-record episode", play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded_episodes += 1
                logging.info(f"✓ Episode {recorded_episodes}/{dataset_config.num_episodes} saved")
                
    except KeyboardInterrupt:
        logging.info("")
        logging.info("Recording interrupted by user")
    except Exception as e:
        logging.error("")
        logging.error("=" * 80)
        logging.error("Recording Error")
        logging.error("=" * 80)
        logging.error(f"Error: {str(e)}")
        logging.error("=" * 80)
        import traceback
        traceback.print_exc()
        raise
    finally:
        log_say("Stop recording", play_sounds, blocking=True)

        # Cleanup
        if dataset:
            dataset.finalize()

        if robot.is_connected:
            robot.disconnect()
            
        if teleop and teleop.is_connected:
            teleop.disconnect()
            
        if remote_policy is not None:
            remote_policy.close()

        if not is_headless() and listener:
            listener.stop()

        # Push to hub if requested
        if dataset_config.push_to_hub and dataset:
            logging.info("")
            logging.info("Pushing dataset to Hugging Face Hub...")
            dataset.push_to_hub(tags=dataset_config.tags, private=dataset_config.private)
            logging.info("✓ Dataset pushed successfully")

        logging.info("")
        log_say("Exiting", play_sounds)
        logging.info("")
        
    return dataset


@parser.wrap()
def run(cfg: RecordConfig) -> None:
    """Parse config and run the recording client."""
    register_third_party_plugins()

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    record(
        robot=robot,
        dataset_config=cfg.dataset,
        remote_policy_config=cfg.remote_policy,
        teleop=teleop,
        display_data=cfg.display_data,
        play_sounds=cfg.play_sounds,
        resume=cfg.resume,
    )


if __name__ == "__main__":
    run()
