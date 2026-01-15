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
Interactive robot client with a simple UI:
- shows current task
- shows recent robot state values
- shows camera frames
"""

import logging
import threading
import time
from dataclasses import asdict
from pprint import pformat
from typing import Any

import draccus
import numpy as np
import torch

from .configs import RobotClientConfig
from .constants import SUPPORTED_ROBOTS
from .helpers import visualize_action_queue_size
from .robot_client import RobotClient


class SharedState:
    def __init__(self, task: str):
        self._lock = threading.Lock()
        self._task = task
        self._observation: dict[str, Any] | None = None

    def set_task(self, task: str) -> None:
        with self._lock:
            self._task = task

    def set_observation(self, obs: dict[str, Any]) -> None:
        with self._lock:
            self._observation = obs

    def snapshot(self) -> tuple[str, dict[str, Any] | None]:
        with self._lock:
            return self._task, self._observation


def _to_bgr_uint8(value: Any) -> np.ndarray | None:
    if not isinstance(value, (torch.Tensor, np.ndarray)):
        return None
    arr = value.cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if arr.ndim != 3:
        return None
    if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 3:
        arr = arr[..., ::-1]
    return arr


def _resize_to_fit(img: np.ndarray, cv2, max_w: int = 960, max_h: int = 720) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale == 1.0:
        return img
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _state_lines(obs: dict[str, Any], max_items: int = 8) -> list[str]:
    lines: list[str] = []
    for key, value in obs.items():
        if key == "task":
            continue
        if isinstance(value, (torch.Tensor, np.ndarray)):
            arr = value
            if isinstance(arr, torch.Tensor):
                arr = arr.detach().cpu().numpy()
            if arr.ndim >= 2 and arr.shape[-1] in (1, 3):
                continue
            if arr.ndim == 1 and arr.size <= 8:
                vals = ", ".join(f"{v:.3f}" for v in arr.tolist())
                lines.append(f"{key}: [{vals}]")
            else:
                lines.append(f"{key}: shape={tuple(arr.shape)}")
        elif isinstance(value, (int, float, str)):
            lines.append(f"{key}: {value}")
        else:
            lines.append(f"{key}: {type(value).__name__}")
        if len(lines) >= max_items:
            break
    return lines


def ui_loop(state: SharedState, stop_event: threading.Event, ui_fps: int = 10) -> None:
    try:
        import cv2  # type: ignore
    except Exception:
        print("OpenCV not available; UI will not be shown.")
        return

    delay_ms = max(1, int(1000 / ui_fps))
    while not stop_event.is_set():
        task, obs = state.snapshot()
        panel = np.zeros((360, 720, 3), dtype=np.uint8)
        cv2.putText(panel, "Task:", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        cv2.putText(panel, task[:64], (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)
        cv2.putText(panel, "State:", (10, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        y = 130
        if obs:
            for key, value in obs.items():
                img = _to_bgr_uint8(value)
                if img is None:
                    continue
                img = _resize_to_fit(img, cv2)
                name = key.split(".")[-1]
                cv2.imshow(f"camera:{name}", img)

            for line in _state_lines(obs):
                cv2.putText(panel, line[:80], (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
                y += 24
        else:
            cv2.putText(panel, "No observation yet.", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        cv2.imshow("status", panel)

        if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
            stop_event.set()
            break

    cv2.destroyAllWindows()


def task_input_loop(state: SharedState, stop_event: threading.Event) -> None:
    print(
        "Task input ready. Type a new task and press Enter.\n"
        "Commands: /task (prompt), /show (print current task), /quit (stop client)"
    )
    while not stop_event.is_set():
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line in {"/quit", "/exit"}:
            stop_event.set()
            break
        if line in {"/show"}:
            print(f"Current task: {state.snapshot()[0]}")
            continue
        if line in {"/task", "/r"}:
            try:
                line = input("New task> ").strip()
            except EOFError:
                break
        if line:
            state.set_task(line)
            print(f"Updated task: {line}")


def control_loop_dynamic_task(
    client: RobotClient, state: SharedState, stop_event: threading.Event, verbose: bool = False
) -> None:
    client.start_barrier.wait()
    client.logger.info("Control loop thread starting (dynamic task + UI)")

    while client.running and not stop_event.is_set():
        control_loop_start = time.perf_counter()
        if client.actions_available():
            client.control_loop_action(verbose)

        if client._ready_to_send_observation():
            obs = client.control_loop_observation(state.snapshot()[0], verbose)
            if obs is not None:
                state.set_observation(obs)

        time.sleep(max(0, client.config.environment_dt - (time.perf_counter() - control_loop_start)))


@draccus.wrap()
def async_client_task_prompt_ui(cfg: RobotClientConfig):
    logging.info(pformat(asdict(cfg)))

    if cfg.robot.type not in SUPPORTED_ROBOTS:
        raise ValueError(f"Robot {cfg.robot.type} not yet supported!")

    initial_task = cfg.task.strip()
    if not initial_task:
        initial_task = input("Initial task> ").strip()

    state = SharedState(initial_task)
    client = RobotClient(cfg)

    if client.start():
        client.logger.info("Starting action receiver thread...")
        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
        action_receiver_thread.start()

        stop_event = threading.Event()
        input_thread = threading.Thread(target=task_input_loop, args=(state, stop_event), daemon=True)
        input_thread.start()

        ui_thread = threading.Thread(target=ui_loop, args=(state, stop_event), daemon=True)
        ui_thread.start()

        try:
            control_loop_dynamic_task(client, state, stop_event)
        finally:
            client.stop()
            stop_event.set()
            action_receiver_thread.join()
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)
            client.logger.info("Client stopped")


if __name__ == "__main__":
    async_client_task_prompt_ui()
