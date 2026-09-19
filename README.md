# 2030-pi-05-sft

基于 **π0.5** 的 PiperX 双臂机器人监督微调（SFT）工程。使用 LoRA 微调语言与动作模块，直接读取 PiperX / Quest 采集器输出的 **LeRobot v3.0 MP4 + Parquet 数据**。

本仓库从实际运行的训练工程导出。导出时，`train.py`、`dataset.py`、`runtime.py` 和 `progress.py` 与本次训练启动记录中的 SHA256 一致。迁移版调整了安装入口及数据、权重的默认路径。

## 功能

- 直接读取原始视频和 Parquet，无需转换为 v2.1。
- 支持三路 RGB 观测和双臂 14 维状态、动作。
- 每轮打乱帧级样本，保留动作窗口内部的时间顺序。
- 支持单卡梯度累积和单机双卡训练。
- 每轮保存独立 checkpoint，保留模型、优化器状态和归一化统计。
- 自动更新 Markdown 进度文档，逐步记录 loss、梯度范数和学习率。

## 示例任务

将架子上的物品转移到左侧黑色托盘：右臂先把右侧玻璃瓶移到左侧酒精灯旁，随后左臂依次转移酒精灯、玻璃瓶、白色架子，按靠近机械臂到远离机械臂的顺序摆放。

统一任务指令由配置中的 `task_override` 指定：

> Move the objects to the black tray on the left. First, the right arm moves the right-side glass bottle beside the silver alcohol lamp on the left. Then, the left arm transfers the lamp, bottle, and white rack in that order, arranging them from nearest to farthest from the robot.

## 数据格式

| 数据项 | 本次实验 |
| --- | --- |
| 示范轨迹 | 122 条 |
| 训练帧数 | 176,283 帧 |
| 采样频率 | 30 FPS |
| RGB 视频 | 366 个，每条轨迹三路 |
| 原始状态 / 动作维度 | 14 |
| 数据划分 | 全部用于训练，不设验证集 |
| 带无效标记的帧 | 4,965 帧，保留并报告 |

数据集：[SheaWang/2030-pi05-sft（ModelScope）](https://modelscope.cn/datasets/SheaWang/2030-pi05-sft)。访问权限以该仓库设置为准。数据、基础权重和 checkpoint 需单独准备。

读取器支持当前采集器的**每个 `episode_*` 目录包含一条轨迹**的结构：

```text
data/piper_x_quest3s_dataset_20260919/data/
├── episode_.../
│   ├── capture.json
│   ├── meta/
│   │   ├── info.json
│   │   ├── tasks.parquet
│   │   └── episodes/...
│   ├── data/.../*.parquet
│   └── videos/...
└── episode_.../
```

实际文件路径由元数据解析。该读取器尚不支持任意多个 episode 共用视频或 Parquet 分片的 v3 数据布局。

三路相机为 `cam_high`、`cam_left_wrist`、`cam_right_wrist`。状态和动作顺序为：**左臂 J1–J6、左夹爪、右臂 J1–J6、右夹爪**；关节单位为弧度，夹爪开度单位为米。

每个样本输入当前图像和状态，监督当前帧起连续 16 帧动作。轨迹末尾重复最后一个动作补齐，窗口不会跨轨迹。关节目标在归一化前转换为相对当前状态的差值，夹爪保留绝对开度；模型内部将动作维度补齐到 32。

完整轨迹参与训练，但每次输入不包含整段轨迹历史。无效标记不会自动触发删帧或切段；不完整轨迹会被列出，损坏文件、非有限数值和视频对齐错误会阻止检查通过。

## 环境要求

目标环境为 **Linux x86_64、Python 3.11、NVIDIA GPU**，使用 CUDA 12 版本的 JAX。

| 组件 | 固定版本 |
| --- | --- |
| JAX / JAXlib | 0.5.3 |
| Flax | 0.10.2 |
| PyTorch | 2.7.1 |
| Orbax Checkpoint | 0.11.13 |
| NumPy | 1.26.4 |
| PyArrow / PyAV | 20.0.0 / 17.0.0 |

完整版本见 [requirements.lock.txt](requirements.lock.txt)。本次双卡配置实测每卡约占用 **33 GiB 显存**；24 GiB 显卡需要减小 batch 等调整。

## 快速开始

以下命令均在仓库根目录执行。

### 1. 安装独立环境

```bash
bash setup.sh
```

也可以使用清华 PyPI 镜像：

```bash
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple bash setup.sh
```

脚本会创建 `.venv`，按固定版本清单安装依赖，并安装随仓库提供的 OpenPI、OpenPI Client 和 LeRobot 源码包。已有兼容 Python 3.11 环境时，可以复用：

```bash
PIPERX_NATIVE_PYTHON=/absolute/path/to/python bash setup.sh --reuse
```

检查环境导入：

```bash
source runtime/environment.sh
"$PIPERX_NATIVE_PYTHON" tools/doctor.py
```

默认使用 CPU 检查；加 `--gpu` 可检查配置中的 GPU 是否可见，不执行训练。迁移包不包含驱动库或原服务器的驱动兼容补丁，新机器需具备兼容的驱动。

### 2. 准备数据和基础权重

复制配置：

```bash
cp config.example.json config.local.json
```

修改以下路径，保留其他配置字段：

```json
{
  "source_root": "./data/piper_x_quest3s_dataset_20260919/data",
  "base_params": "./weights/pi05_base/params"
}
```

相对路径以配置文件所在目录为基准。`source_root` 应指向包含所有 `episode_*` 的父目录；`base_params` 指向基础权重的 `params` 目录。

基础权重使用 **pi05_base**，参考 [OpenPI 官方权重说明](https://github.com/Physical-Intelligence/openpi#model-checkpoints)。还需 tokenizer 等资源缓存，可允许首次联网获取，或复制已有的资源缓存。默认缓存目录为 `cache/huggingface` 和 `cache/openpi`，也可通过 `HF_HOME`、`OPENPI_DATA_HOME` 指定。

### 3. 检查数据并计算归一化统计

```bash
bash run.sh scan --config config.local.json

source runtime/environment.sh
"$PIPERX_NATIVE_PYTHON" tools/check_parallel.py \
  --config config.local.json --workers 8

bash run.sh stats --config config.local.json

"$PIPERX_NATIVE_PYTHON" tools/check_training_batches.py \
  --config config.local.json
```

检查覆盖元数据、数值、源文件哈希、完整视频解码、帧数和时间对齐，以及边界随机读取。归一化统计从本次训练数据和实际动作窗口重新计算。最后的 batch 检查验证图像、状态、动作、prompt 和末尾补齐规则，不更新模型。

更换数据后应重新执行这些步骤。

### 4. 启动训练

```bash
mkdir -p logs
nohup bash run.sh train \
  --config config.local.json \
  --experiment full122_v3 \
  > logs/full122_v3.log 2>&1 < /dev/null &
```

查看日志：

```bash
tail -f logs/full122_v3.log
```

同名实验目录存在时，脚本会拒绝新建训练，避免覆盖结果。新实验请使用不同的 `--experiment` 名称。

## 训练参数

| 参数 | 示例配置 | 含义 |
| --- | --- | --- |
| `gpu_ids` | `"0,1"` | 使用两张 GPU |
| `epochs` | 20 | 每轮遍历全部真实训练帧 |
| `batch_size` | 48 | 总 batch，每卡 24 |
| `gradient_accumulation_steps` | 1 | 双卡当前要求为 1 |
| `language_rank` | 32 | 语言模块 LoRA rank，alpha 同为 32 |
| `action_rank` | 64 | 动作模块 LoRA rank，alpha 同为 64 |
| `learning_rate` | `2.5e-5` | 预热后的峰值，余弦降至 `2.5e-6` |
| `warmup_epochs` | 0.1 | 本次数据对应 368 次更新 |
| `action_horizon` | 16 | 每个观测对应的动作窗口 |
| `image_augmentation` | `false` | 关闭随机图像增强 |
| `num_workers` | 4 | 数据读取进程数 |
| `seed` | 42 | 可复现的每轮样本排列 |

`task_override` 会统一覆盖采集数据中的任务文字，更换任务时应同步修改。

优化器为 AdamW，梯度裁剪阈值为 1.0，关闭 EMA。冻结规则沿用当前 OpenPI LoRA 配置，视觉分支等参数也参与训练。增大 LoRA rank 会增加容量，但不保证任务成功或必然过拟合。

本次数据每轮更新数为 `ceil(176283 / 48) = 3673`，20 轮共 **73,460 次更新**。每轮最后的 27 个真实样本会补齐到双卡可分配的数量；补齐样本权重为零，不影响 loss，每个真实帧每轮恰好参与一次。

单卡可设置 `gpu_ids: "0"`、`gradient_accumulation_steps: 2`，总 batch 仍为 48；实际显存是否足够需在目标机器确认。

## 进度文档与 checkpoint

| 文件 | 内容 |
| --- | --- |
| `reports/训练进度.md` | 阶段、轮次、步数、loss、学习率和预计剩余时间 |
| 工程父目录下的 `π0.5训练进度.md` | 同一进度文档，便于从文件管理器打开 |
| 实验目录下的 `progress.md` | 当前实验进度 |
| 实验目录下的 `metrics.jsonl` | 每步 loss、裁剪前梯度范数、学习率、真实样本数及时间 |
| 实验目录下的 `run.json` | 配置、数据和源码指纹 |

文档每 30 秒刷新，并在每 10 步、保存模型和退出时更新。预计剩余时间包含数据读取，最初编译阶段暂不估计。日志中的 `update_seconds` 只包含拿到 batch 后的传输和模型更新时间，不等于完整训练吞吐。

每完成一轮保存一个独立 checkpoint，保留所有轮次：

```text
checkpoints/pi05_piperx_native_v3_lora/full122_v3/
├── run.json
├── metrics.jsonl
├── progress.md
├── 3673/       # 第 1 轮
│   ├── params/
│   ├── train_state/
│   └── assets/
├── 3673.ready.json
├── 7346/       # 第 2 轮
└── ...
```

正常完成 20 轮会保留 20 个轮末模型。`.ready.json` 在对应 checkpoint 写入完成后生成。使用 `--stop-after-step` 主动暂停时，也会额外保存该步状态。异常终止时以最近完整 checkpoint 为准，最后一条 loss 日志不一定对应已保存的模型。

## 断点续训与迁移

在同一工程中，使用相同配置和实验名称恢复：

```bash
bash run.sh train \
  --config config.local.json \
  --experiment full122_v3 \
  --resume
```

恢复会核对数据、源码、配置和归一化统计，同时恢复模型、Adam 状态及优化器步数，从确定的下一批样本继续。修改 rank、数据或训练计划后应新建实验。

本包可用于在新机器准备环境并重新训练。跨机器继续已有实验，还需迁移 checkpoint、归一化统计和训练记录，并处理绝对路径及指纹校验；仅复制模型参数不等于恢复原训练过程。

## 验证范围

原训练工程已完成真实数据检查、短程单卡训练、checkpoint 保存恢复和离线推理验证，随后启动双卡全量训练。迁移包已通过 Python / Shell 语法检查、CPU 环境导入及三个本地依赖包的 wheel 构建检查。

**尚未在全新服务器完整验证从零安装。** 不同机器的驱动、显存和依赖下载条件需要实际确认，详细记录见 [VALIDATION.json](VALIDATION.json)。

当前实验用于充分拟合一个固定任务的示范，不设置验证集。训练 loss 下降表示示范拟合改善，不能直接解释为动作误差百分比或实机成功率；实际执行效果需另行评估。

## 目录与来源

```text
├── train.py                 # 训练、统计、检查和推理入口
├── dataset.py               # 原生 v3 数据读取、采样与 batch 补齐
├── runtime.py               # 路径、源码指纹和进程级模型配置
├── progress.py              # Markdown / JSON 进度文档
├── config.example.json      # 本次实验的参考配置
├── setup.sh / run.sh        # 环境准备与运行入口
├── requirements.lock.txt    # 固定 Python 依赖版本
├── tools/                   # 数据检查与环境检查工具
├── runtime/openpi/          # 所使用的 OpenPI 源码快照
└── runtime/vendor/lerobot/   # 满足 OpenPI 导入需要的 LeRobot 源码
```

OpenPI 快照基于提交 `215abfb217dbac7d5f1273282331b9b1866c0479`，包含原工程对梯度累积、恢复逻辑及配置的修改。LeRobot 依赖用于兼容 OpenPI 导入；本工程实际使用 `dataset.py` 读取 v3 数据。

第三方来源、修改说明及许可证见 [THIRD_PARTY.md](THIRD_PARTY.md)。新增项目代码的许可证尚待仓库拥有者选定；模型权重遵循其各自的使用条款。
