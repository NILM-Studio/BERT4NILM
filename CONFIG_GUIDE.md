# BERT4NILM 配置文件使用指南

## 概述

本项目已支持使用配置文件（JSON 格式）进行训练，适用于 SLURM 集群环境。

## 文件说明

- `config.json`：主配置文件模板
- `submit_slurm.sh`：SLURM 提交脚本示例

## 使用方法

### 方法 1：使用配置文件训练（推荐）

```bash
# 使用默认配置文件
python train.py --config config.json

# 使用自定义配置文件
python train.py --config config_uk_dale.json
```

### 方法 2：交互式训练（本地开发）

```bash
# 不指定配置文件，使用交互式输入
python train.py
```

### 方法 3：在 SLURM 集群上训练

```bash
# 提交作业
sbatch submit_slurm.sh

# 查看作业状态
squeue -u $USER
```

## 配置文件格式

配置文件使用 JSON 格式，包含以下参数：

```json
{
  "paths": {
    "raw_dataset_root": "E:\\datasets\\NILM\\",
    "experiment_root": "experiments"
  },
  "device": "cuda:0",
  "dataset_code": "csv_dataset",
  "appliance_names": ["fridge", "microwave", "kettle"],
  "num_epochs": 100,
  "batch_size": 128,
  "window_size": 480,
  "window_stride": 240,
  "sampling": "6s",
  "validation_size": 0.1,
  "normalize": "mean",
  "denom": 2000,
  "drop_out": 0.1,
  "mask_prob": 0.25,
  "optimizer": "adam",
  "lr": 1e-4,
  "weight_decay": 0.0,
  "enable_lr_schedule": false,
  "decay_step": 100,
  "gamma": 0.1,
  "cutoff": {
    "aggregate": 6000,
    "fridge": 600,
    "microwave": 2000,
    "kettle": 3100
  },
  "threshold": {
    "fridge": 50,
    "microwave": 200,
    "kettle": 2000
  },
  "min_on": {
    "fridge": 10,
    "microwave": 2,
    "kettle": 2
  },
  "min_off": {
    "fridge": 2,
    "microwave": 5,
    "kettle": 0
  },
  "c0": {
    "fridge": 1e-6,
    "microwave": 1.0,
    "kettle": 1.0
  }
}
```

## 参数说明

### 路径参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| paths.raw_dataset_root | string | "E:\\datasets\\NILM\\" | 数据集根目录路径 |
| paths.experiment_root | string | "experiments" | 实验结果保存目录 |

### 基础参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| device | string | "cpu" | 计算设备（"cpu" 或 "cuda:0"） |
| dataset_code | string | "csv_dataset" | 数据集类型（"csv_dataset", "uk_dale", "redd_lf"） |
| appliance_names | list | ["fridge", "microwave"] | 电器名称列表 |
| num_epochs | int | 100 | 训练轮数 |
| batch_size | int | 128 | 批量大小 |
| window_size | int | 480 | 窗口大小（时间步数） |
| window_stride | int | 240 | 窗口步长 |
| sampling | string | "6s" | 采样频率 |
| validation_size | float | 0.1 | 验证集比例 |

### 模型参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| normalize | string | "mean" | 归一化方式（"mean" 或 "minmax"） |
| denom | int | 2000 | 分母参数 |
| drop_out | float | 0.1 | Dropout 比率 |
| mask_prob | float | 0.25 | BERT 掩码概率 |

### 优化器参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| optimizer | string | "adam" | 优化器类型（"adam", "sgd", "adamw"） |
| lr | float | 1e-4 | 学习率 |
| weight_decay | float | 0.0 | 权重衰减 |
| enable_lr_schedule | bool | false | 是否启用学习率调度 |
| decay_step | int | 100 | 学习率衰减步数 |
| gamma | float | 0.1 | 学习率衰减系数 |

### 电器参数

每个电器需要配置以下参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| cutoff | dict | {} | 功率上限（瓦特） |
| threshold | dict | {} | 判断开启的功率阈值（瓦特） |
| min_on | dict | {} | 最小开启时间（时间步数） |
| min_off | dict | {} | 最小关闭时间（时间步数） |
| c0 | dict | {} | L1 损失权重 |

## 不同数据集的配置示例

### CSV 数据集

```json
{
  "device": "cuda:0",
  "dataset_code": "csv_dataset",
  "appliance_names": ["fridge", "microwave", "kettle"],
  "window_stride": 240,
  "house_indicies": [1]
}
```

### UK-DALE 数据集

```json
{
  "device": "cuda:0",
  "dataset_code": "uk_dale",
  "appliance_names": ["kettle", "fridge", "washing_machine"],
  "window_stride": 240,
  "house_indicies": [1, 2, 3, 4, 5]
}
```

### REDD 数据集

```json
{
  "device": "cuda:0",
  "dataset_code": "redd_lf",
  "appliance_names": ["refrigerator", "washer_dryer"],
  "window_stride": 120,
  "house_indicies": [1, 2, 3, 4, 5, 6]
}
```

## SLURM 脚本配置

### 修改 submit_slurm.sh

根据您的集群配置修改以下参数：

```bash
#SBATCH --job-name=bert4nilm
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
```

### 常用 SLURM 命令

```bash
# 提交作业
sbatch submit_slurm.sh

# 查看作业状态
squeue -u $USER

# 取消作业
scancel <job_id>

# 查看作业日志
tail -f bert4nilm_<job_id>.out
```

## 输出结果

训练完成后，结果保存在：

```
experiments/<dataset_code>/<appliance_names>/
├── best_acc_model.pth      # 最佳模型权重
└── test_result.json        # 测试结果
```

## 故障排除

### 问题：配置文件未找到

```
FileNotFoundError: [Errno 2] No such file or directory: 'config.json'
```

**解决**：确保配置文件路径正确，使用绝对路径或相对路径

### 问题：参数类型错误

```
TypeError: '...' object is not callable
```

**解决**：检查配置文件中的参数类型是否正确（字符串、数字、布尔值等）

### 问题：SLURM 作业失败

```
slurmstepd: error: Job <job_id> failed
```

**解决**：查看错误日志
```bash
cat bert4nilm_<job_id>.err
```

## 高级用法

### 多个配置文件

为不同的实验创建多个配置文件：

```bash
# 配置文件 1
python train.py --config config_exp1.json

# 配置文件 2
python train.py --config config_exp2.json

# 配置文件 3
python train.py --config config_exp3.json
```

### 批量提交 SLURM 作业

创建批量提交脚本：

```bash
#!/bin/bash
for config in config_*.json; do
    sbatch --export=ALL submit_slurm.sh
done
```

## 联系方式

如有问题，请参考：
- 项目 README.md
- 原始论文：BERT4NILM: A Bidirectional Transformer Model for Non-Intrusive Load Monitoring
