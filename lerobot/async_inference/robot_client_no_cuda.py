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
CPU-safe RobotClient for async inference.

This wraps RobotClient.receive_actions to force CPU deserialization when the
client machine has no CUDA.
"""

import io
import logging
import os
import threading
from dataclasses import asdict
from pathlib import Path
from pprint import pformat

import draccus
import numpy as np
import torch

from .configs import RobotClientConfig
from .constants import SUPPORTED_ROBOTS
from .robot_client import RobotClient
from .helpers import visualize_action_queue_size


class RobotClientNoCuda(RobotClient):
    def __init__(self, config: RobotClientConfig):
        super().__init__(config)
        self._preview_dir = os.getenv("LEROBOT_PREVIEW_DIR", "").strip() or None
        self._preview_every = max(1, int(os.getenv("LEROBOT_PREVIEW_EVERY", "1")))
        self._preview_overwrite = os.getenv("LEROBOT_PREVIEW_OVERWRITE", "true").strip().lower() != "false"
        self._preview_step = 0

    def receive_actions(self, verbose: bool = False):
        if not torch.cuda.is_available():
            original_load_from_bytes = torch.storage._load_from_bytes

            def _load_from_bytes_cpu(b):
                return torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)

            torch.storage._load_from_bytes = _load_from_bytes_cpu
            try:
                return super().receive_actions(verbose)
            finally:
                torch.storage._load_from_bytes = original_load_from_bytes
        return super().receive_actions(verbose)

    def control_loop_observation(self, task: str, verbose: bool = False):
        raw_observation = super().control_loop_observation(task, verbose)
        if self._preview_dir and self._preview_step % self._preview_every == 0:
            self._save_preview_images(raw_observation)
        self._preview_step += 1
        return raw_observation

    def _save_preview_images(self, raw_observation):
        try:
            import cv2  # type: ignore
        except Exception:
            self.logger.warning("OpenCV not available; preview images will not be saved.")
            return

        preview_dir = Path(self._preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)

        for key, value in raw_observation.items():
            if not isinstance(value, (torch.Tensor, np.ndarray)):
                continue
            arr = value.cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
            if arr.ndim != 3:
                continue
            if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                arr = arr.astype(np.float32)
                if arr.max() <= 1.0:
                    arr = arr * 255.0
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            if arr.shape[-1] == 3:
                arr = arr[..., ::-1]

            name = key.split(".")[-1]
            if self._preview_overwrite:
                out_path = preview_dir / f"latest_{name}.jpg"
            else:
                out_path = preview_dir / f"{name}_{self._preview_step:06d}.jpg"
            cv2.imwrite(str(out_path), arr)


@draccus.wrap()
def async_client(cfg: RobotClientConfig):
    logging.info(pformat(asdict(cfg)))

    if cfg.robot.type not in SUPPORTED_ROBOTS:
        raise ValueError(f"Robot {cfg.robot.type} not yet supported!")

    client = RobotClientNoCuda(cfg)

    if client.start():
        client.logger.info("Starting action receiver thread...")

        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
        action_receiver_thread.start()

        try:
            client.control_loop(task=cfg.task)
        finally:
            client.stop()
            action_receiver_thread.join()
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)
            client.logger.info("Client stopped")


if __name__ == "__main__":
    async_client()
