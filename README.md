# LeRobot SO101 - 数据采集与本地推理

这是一个用于 SO101 数据采集与本地推理的独立工作区。

## 快速开始

```bash
cd /path/to/lerobot_record_inference
```

## 数据采集（录制）

```bash
python lerobot/scripts/lerobot_record.py \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower \
  --robot.cameras='{"top":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30},"side":{"type":"opencv","index_or_path":4,"width":640,"height":480,"fps":30},"wrist":{"type":"opencv","index_or_path":2,"width":640,"height":480,"fps":30}}' \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=leader \
  --dataset.repo_id=local/pick_red_cylinder \
  --dataset.root=dataset/pick_red_cylinder \
  --dataset.single_task="Pick up a tall red cylinder." \
  --dataset.fps=30 \
  --dataset.num_episodes=100 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=15 \
  --dataset.push_to_hub=false \
  --display_data=true
```

### 需要修改的字段

- `--robot.port`: 从臂端口（查看 `/dev/ttyACM*`）。
- `--teleop.port`: 主臂端口（查看 `/dev/ttyACM*`）。
- `--robot.cameras`: 摄像头 `index_or_path`。
- `--dataset.repo_id`: 数据集名称（local/...）。
- `--dataset.root`: 数据集保存路径。
- `--dataset.single_task`: 任务描述（英文）。

## 本地推理（policy_server + UI client）

终端 1：
```bash
python -m lerobot.async_inference.policy_server \
  --host=127.0.0.1 \
  --port=5555 \
  --fps=30 \
  --inference_latency=0.033 \
  --obs_queue_timeout=1
```

终端 2：
```bash
python -m lerobot.async_inference.robot_client_task_prompt_ui \
  --server_address=127.0.0.1:5555 \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower \
  --robot.cameras='{"top":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30},"side":{"type":"opencv","index_or_path":4,"width":640,"height":480,"fps":30},"wrist":{"type":"opencv","index_or_path":2,"width":640,"height":480,"fps":30}}' \
  --policy_type=pi05 \
  --pretrained_name_or_path=/path/to/lerobot_record_inference/pi05_so101_cube_only_3ep \
  --policy_device=cuda \
  --actions_per_chunk=50 \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name=weighted_average \
  --debug_visualize_queue_size=True
```

### 需要修改的字段

- `--robot.port`: 从臂端口。
- `--robot.cameras`: 摄像头 `index_or_path`。
- `--pretrained_name_or_path`: 本地模型路径（本工作区内）。
- `--policy_device`: 无 GPU 时设置为 `cpu`。

## 加载新模型前：更新 PaliGemma 路径

当模型目录搬迁或 `paligemma` 位置变化时，需要更新各模型目录里的
`policy_preprocessor.json` 的 `tokenizer_name` 字段。

```bash
python lerobot/scripts/update_tokenizer_path.py \
  --paligemma /path/to/lerobot_record_inference/paligemma \
  --model /path/to/model 
```

## 备注

- 模型与数据集保存在本地，已在 git 中忽略。
- 更新 tokenizer 路径可用 `lerobot/scripts/update_tokenizer_path.py`。
