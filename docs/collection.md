第一步：环境准备

```bash
conda activate lerobot
cd /home/baai/桌面/lerobot_record_inference
```

第二步：运行数据录制

```bash
python lerobot/scripts/lerobot_record.py \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower \
    --robot.cameras='{
        "top":   {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
        "side":  {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30},
        "wrist": {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}
    }' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=leader \
    --dataset.repo_id=local/pick_red_cylinder \
    --dataset.root=dataset/pick_up_dark_green_cuboid_0 \
    --dataset.single_task="Pick up a dark green cuboid." \
    --dataset.fps=30 \
    --dataset.num_episodes=100 \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=15 \
    --dataset.push_to_hub=false \
    --display_data=true
```

Pick up an orange cube.
Pick up a short yellow cylinder.
Pick up a tall red cylinder.
Pick up a dark green cuboid.
Pick up a half-height green cuboid.
Pick up a purple triangular prism.
Pick up a purple elongated cuboid

    --robot.port:

        /dev/ttyACM1 是从臂（Follower）的端口。

        /dev/ttyACM0 是主臂（Leader）的端口。

        注意：如果运行报错找不到设备，请使用 ls /dev/ttyACM* 查看实际端口号并替换。

    "index_or_path":

        这是摄像头的 ID（0, 2, 4 等）。

        注意：如果画面不对或打不开，请尝试更换数字。

    --dataset.single_task:

        修改为当前任务的英文描述。

    --dataset.root:

        数据存储的本地路径，建议确认拼写。

第三步：上传数据集至 ModelScope

录制完成后，将数据集备份/上传到 ModelScope。
1. 登录 ModelScope
```bash

# 请将 [Your-Modelscope-Token] 替换为你从 ModelScope 官网获取的 SDK 令牌
modelscope login --token YOUR_ACTUAL_TOKEN
```

2. 快速上传整个文件夹
```bash

# 用法：modelscope upload [仓库ID] [本地路径] --repo-type dataset
# 修改示例：
modelscope upload hym227/so101_cube_3cam ./pick_orange_cube_3 --repo-type dataset
```

```bash
python - <<'PY'
from pathlib import Path
from lerobot.datasets.aggregate import aggregate_datasets

# ===== 1. 子数据集路径（唯一真源）=====
roots = [
    Path("pick_orange_cube"),
    Path("pick_red_cylinder"),
    Path("pick_yellow_cylinder"),
]

# ===== 2. 由路径自动生成 repo_ids =====
# 规则：local/<目录名>
repo_ids = [f"local/{root.name}" for root in roots]

# ===== 3. 聚合 repo =====
aggr_repo_id = "local/pick"
aggr_root = Path("./pick")

# ===== 4. 聚合 =====
aggregate_datasets(
    repo_ids=repo_ids,
    aggr_repo_id=aggr_repo_id,
    roots=roots,
    aggr_root=aggr_root,
)

print("done")
PY
```
