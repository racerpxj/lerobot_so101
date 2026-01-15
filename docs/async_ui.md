# 本地推理（SO101 + 本地模型，带简易 UI）

在本工作区根目录下运行。需要两个终端。

```bash
cd /home/baai/桌面/lerobot_record_inference
```

## 终端 1：启动本地推理服务

```bash
python -m lerobot.async_inference.policy_server \
  --host=127.0.0.1 \
  --port=5555 \
  --fps=30 \
  --inference_latency=0.033 \
  --obs_queue_timeout=1
```

## 终端 2：启动机器人客户端（带 UI，可动态修改 task）

```bash
python -m lerobot.async_inference.robot_client_task_prompt_ui \
  --server_address=127.0.0.1:5555 \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower \
  --robot.cameras='{"top":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30},"side":{"type":"opencv","index_or_path":4,"width":640,"height":480,"fps":30},"wrist":{"type":"opencv","index_or_path":2,"width":640,"height":480,"fps":30}}' \
  --policy_type=pi05 \
  --pretrained_name_or_path=/home/baai/桌面/lerobot_record_inference/pi05_so101_cube_only_3ep \
  --policy_device=cuda \
  --actions_per_chunk=50 \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name=weighted_average \
  --debug_visualize_queue_size=True
```

## 界面说明

- `status` 窗口显示当前 task 和状态摘要
- `camera:*` 窗口显示各相机图像
- 在任意窗口按 `q` 可退出

## 交互说明

启动后会提示输入初始 task（也可以通过 `--task` 直接传入）。运行中可随时更新：

- 直接输入新 task 回车：立即更新
- `/task` 或 `/r`：重新输入 task
- `/show`：打印当前 task
- `/quit`：退出客户端
