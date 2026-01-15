#!/usr/bin/env python3


from typing import Dict
import time
import os
import json
from datetime import datetime

import cv2
from lerobot.robots import so101_follower


# ==================== 数据保存工具 ====================

def save_step_data(
    base_dir: str,
    obs: Dict[str, float],
    front_img,
    wrist_img,
    prefix: str = "step"
):
    """保存一次 step 的状态和图像"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    step_dir = os.path.join(base_dir, f"{prefix}_{timestamp}")
    os.makedirs(step_dir, exist_ok=True)

    with open(os.path.join(step_dir, "observation.json"), "w") as f:
        json.dump(obs, f, indent=2)

    if front_img is not None:
        cv2.imwrite(os.path.join(step_dir, "front.jpg"), front_img)
    if wrist_img is not None:
        cv2.imwrite(os.path.join(step_dir, "wrist.jpg"), wrist_img)

    print(f"📦 数据已保存到 {step_dir}")


# ==================== SO101 控制器 ====================

class SO101Controller:

    def __init__(
        self,
        port: str = "/dev/tty.usbmodem5AB01587521",
        robot_id: str = "follower_arm_01",
        data_dir: str = "so101_logs"
    ):
        self.port = port
        self.robot_id = robot_id
        self.data_dir = data_dir

        config = so101_follower.SO101FollowerConfig(
            port=port,
            id=robot_id
        )
        self.robot = so101_follower.SO101Follower(config)

        self.bus = None
        self.motors = None

        self.front_cam = None
        self.wrist_cam = None

        self.joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]

    # ---------- 生命周期 ----------

    def connect(self):
        self.robot.connect()

        self.bus = self.robot.bus
        self.motors = self.bus.motors

        if not self.motors:
            raise RuntimeError("未检测到电机")

        # 打开相机（你已确认的 index）
        self.front_cam = cv2.VideoCapture(0)
        self.wrist_cam = cv2.VideoCapture(1)

        print(f"✓ 已连接 {len(self.motors)} 个电机")

    def disconnect(self):
        if self.front_cam:
            self.front_cam.release()
        if self.wrist_cam:
            self.wrist_cam.release()
        self.robot.disconnect()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    # ---------- 状态 ----------

    def get_observation(self) -> Dict[str, float]:
        obs = self.robot.get_observation()
        return {k.replace(".pos", ""): v for k, v in obs.items()}

    def capture_and_save(self, prefix="step"):
        obs = self.get_observation()

        front_img = None
        wrist_img = None

        if self.front_cam:
            _, front_img = self.front_cam.read()
        if self.wrist_cam:
            _, wrist_img = self.wrist_cam.read()

        save_step_data(
            base_dir=self.data_dir,
            obs=obs,
            front_img=front_img,
            wrist_img=wrist_img,
            prefix=prefix
        )

    # ---------- 控制 ----------

    def send_action_absolute(self, goal_positions: Dict[str, float]):
        action = {f"{k}.pos": v for k, v in goal_positions.items()}
        try:
            self.robot.send_action(action)
        except Exception as e:
            print(f"⚠️ send_action 失败，fallback sync_write: {e}")
            self._send_action_via_sync_write(goal_positions)

    def send_action_relative(self, actions: Dict[str, float], scale: float = 10.0):
        current = self.get_observation()
        goal = {}
        for joint, act in actions.items():
            goal[joint] = current.get(joint, 0.0) + act * scale
        self.send_action_absolute(goal)

    def _send_action_via_sync_write(self, goal_positions: Dict[str, float]):
        values = {
            name: int(pos)
            for name, pos in goal_positions.items()
            if name in self.motors
        }
        if values:
            self.bus.sync_write("Goal_Position", values)

    def send_zero_action(self):
        self.send_action_absolute(self.get_observation())


# ==================== 测试 1：基础功能 ====================

def test_basic():
    print("=" * 70)
    print("测试 1: 基础功能")
    print("=" * 70)

    with SO101Controller() as ctrl:
        print("\n初始状态：")
        obs = ctrl.get_observation()
        for k, v in obs.items():
            print(f"  {k}: {v:.2f}")

        ctrl.capture_and_save(prefix="before")

        print("\n发送中等幅度动作（明显变化）...")
        action = {
            "shoulder_pan": 0.8,
            "shoulder_lift": -0.5,
            "elbow_flex": 0.6,
            "wrist_flex": 0.4,
            "wrist_roll": -0.4,
            "gripper": 0.2,
        }
        ctrl.send_action_relative(action, scale=20.0)

        time.sleep(2)
        ctrl.capture_and_save(prefix="after")

        print("\n动作完成")


# ==================== 测试 2：手动输入 ====================

def test_manual_action():
    print("=" * 70)
    print("测试 2: 手动输入动作")
    print("=" * 70)

    with SO101Controller() as ctrl:
        obs = ctrl.get_observation()
        for k, v in obs.items():
            print(f"  {k}: {v:.2f}")

        actions = {}
        for joint in ctrl.joint_names:
            val = input(f"{joint} [-1,1] (Enter=0): ").strip()
            actions[joint] = float(val) if val else 0.0

        scale = input("scale (Enter=20): ").strip()
        scale = float(scale) if scale else 20.0

        ctrl.capture_and_save(prefix="manual_before")
        ctrl.send_action_relative(actions, scale)
        time.sleep(2)
        ctrl.capture_and_save(prefix="manual_after")


# ==================== 测试 3：连续控制 ====================

def test_continuous_control():
    print("=" * 70)
    print("测试 3: 连续控制")
    print("=" * 70)

    with SO101Controller() as ctrl:
        for i in range(10):
            print(f"Step {i+1}/10")
            ctrl.send_action_relative(
                {"shoulder_pan": 0.2},
                scale=5.0
            )
            ctrl.capture_and_save(prefix=f"loop_{i}")
            time.sleep(0.5)


# ==================== 主程序 ====================

if __name__ == "__main__":
    print("=" * 70)
    print("SO101 控制器")
    print("=" * 70)
    print("1 - 基础功能测试")
    print("2 - 手动输入测试")
    print("3 - 连续控制测试")

    choice = input("选择测试 [1-3]: ").strip()

    try:
        if choice == "1":
            test_basic()
        elif choice == "2":
            test_manual_action()
        elif choice == "3":
            test_continuous_control()
        else:
            print("无效选项")
    except KeyboardInterrupt:
        print("\n用户中断")
