# CSV数据集使用指南

## 一、CSV文件格式要求

### 文件结构
```
your_dataset_folder/
├── train.csv    # 训练集
├── val.csv      # 验证集（可选）
└── test.csv     # 测试集（可选）
```

### CSV文件格式

#### 必需列
- **第1列**：时间戳（可以是Unix时间戳或datetime格式）
- **第2列**：总功率（aggregate）
- **第3列及之后**：各电器的功率值，列名为电器名字

#### 示例CSV文件

**train.csv**
```csv
timestamp,aggregate,fridge,microwave,kettle
1234567890,450.5,120.3,0,0
1234567896,465.2,125.1,0,0
1234567902,1450.8,118.7,1200.0,0
1234567908,1650.3,122.4,1200.0,300.0
1234567914,450.2,119.8,0,0
```

**或者使用datetime格式**
```csv
time,aggregate,fridge,microwave,kettle
2024-01-01 00:00:00,450.5,120.3,0,0
2024-01-01 00:00:06,465.2,125.1,0,0
2024-01-01 00:00:12,1450.8,118.7,1200.0,0
2024-01-01 00:00:18,1650.3,122.4,1200.0,300.0
2024-01-01 00:00:24,450.2,119.8,0,0
```

### 数据要求

1. **时间戳**：
   - 支持Unix时间戳（秒）
   - 支持datetime格式（YYYY-MM-DD HH:MM:SS）
   - 时间间隔应保持一致（如每6秒一个数据点）

2. **功率值**：
   - 单位：瓦特（W）
   - 数值应为非负数
   - 建议过滤掉小于5W的噪声数据

3. **缺失值**：
   - 尽量避免缺失值
   - 如果有缺失值，程序会自动删除该行

## 二、配置文件设置

### 1. 修改config.py

```python
# 设置CSV数据集的根目录
RAW_DATASET_ROOT_FOLDER = 'E:\datasets\NILM'  # 改为你的CSV文件所在目录
```

### 2. 数据准备

将你的CSV文件放到指定目录：

```
E:\datasets\NILM\
├── train.csv    # 包含 timestamp, aggregate, fridge, microwave, kettle 等列
├── val.csv      # 可选
└── test.csv     # 可选
```

## 三、使用方法

### 方法1：交互式运行

```bash
python train.py
```

运行后会提示输入：

```
Input GPU ID: cpu  # 或 cuda:0
Input r for REDD, u for UK_DALE, c for CSV: c
Loading training data from E:\datasets\NILM\train.csv to detect appliance names...
Automatically detected appliance names: ['fridge', 'microwave', 'kettle']
You can press Enter to confirm or input custom appliance names (comma separated):  # 直接回车确认
Input training epochs: 50
```

**注意**：对于 CSV 数据集，系统会自动从 CSV 文件中读取列名作为电器列表，你可以直接回车确认，也可以输入自定义的电器名称。

**CSV文件格式要求**：
- 必须包含 `aggregate` 列（总功率）
- 可以包含 `timestamp` 或 `time` 列（时间戳）
- 其他列名即为电器名称

### 方法2：命令行参数运行

创建一个运行脚本 `run_csv_training.py`：

```python
from dataset import CSV_Dataset
from dataloader import *
from trainer import *
from config import *
from utils import *
from model import BERT4NILM

import torch
import os

def main():
    # 设置参数
    class Args:
        def __init__(self):
            self.seed = 12345
            self.dataset_code = 'csv_dataset'
            self.validation_size = 0.1
            self.batch_size = 128
            self.house_indicies = [1]
            self.appliance_names = ['fridge', 'microwave', 'kettle']
            self.sampling = '6s'
            self.window_size = 480
            self.window_stride = 120
            self.normalize = 'mean'
            self.denom = 2000
            self.output_size = 3
            self.drop_out = 0.1
            self.mask_prob = 0.25
            self.device = 'cpu'
            self.optimizer = 'adam'
            self.lr = 1e-4
            self.weight_decay = 0.0
            self.momentum = None
            self.decay_step = 100
            self.gamma = 0.1
            self.num_epochs = 50
            self.enable_lr_schedule = False
            
            # 设置电器参数
            self.cutoff = {
                'aggregate': 6000,
                'fridge': 6000,
                'microwave': 6000,
                'kettle': 6000,
            }
            self.threshold = {
                'fridge': 50,
                'microwave': 200,
                'kettle': 2000,
            }
            self.min_on = {
                'fridge': 10,
                'microwave': 2,
                'kettle': 2,
            }
            self.min_off = {
                'fridge': 2,
                'microwave': 5,
                'kettle': 0,
            }
            self.c0 = {
                'fridge': 1e-6,
                'microwave': 1.0,
                'kettle': 1.0,
            }
    
    args = Args()
    
    # 设置随机种子
    import random
    import numpy as np
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 加载数据集
    print('Loading CSV dataset...')
    dataset = CSV_Dataset(args)
    
    # 获取标准化参数
    x_mean, x_std = dataset.get_mean_std()
    stats = (x_mean, x_std)
    
    # 创建模型
    print('Creating model...')
    model = BERT4NILM(args)
    
    # 设置输出目录
    folder_name = '-'.join(args.appliance_names)
    export_root = 'experiments/' + args.dataset_code + '/' + folder_name
    
    # 创建数据加载器
    dataloader = NILMDataloader(args, dataset, bert=True)
    train_loader, val_loader = dataloader.get_dataloaders()
    
    # 创建训练器
    trainer = Trainer(args, model, train_loader, val_loader, stats, export_root)
    
    # 训练模型
    if args.num_epochs > 0:
        print('Starting training...')
        trainer.train()
    
    # 测试模型
    print('Testing model...')
    test_dataset = dataset.get_test_data()
    test_loader = NILMDataloader(args, test_dataset, bert=False)._get_loader(test_dataset)
    rel_err, abs_err, acc, prec, recall, f1 = trainer.test(test_loader)
    
    print('\n' + '='*50)
    print('Test Results:')
    print('='*50)
    print('Mean Accuracy:', acc)
    print('Mean F1-Score:', f1)
    print('Mean Relative Error:', rel_err)
    print('Mean Absolute Error:', abs_err)
    print('='*50)

if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    main()
```

## 四、参数调优建议

### 1. 窗口参数

```python
args.window_size = 480  # 窗口大小（时间步数）
args.window_stride = 120  # 窗口步长
```

- **window_size**：控制模型一次处理的时间长度
  - 较大值（如480）：能捕获更长的依赖关系，但计算量更大
  - 较小值（如240）：计算更快，但可能丢失长期信息

- **window_stride**：控制窗口之间的重叠程度
  - 较小值（如60）：更多重叠，训练数据更多
  - 较大值（如240）：重叠少，训练更快

### 2. 电器参数

根据你的数据特点调整：

```python
# cutoff：功率上限（瓦特）
args.cutoff = {
    'fridge': 600,      # 冰箱通常不超过600W
    'microwave': 2000,  # 微波炉通常不超过2000W
    'kettle': 3100,     # 水壶通常不超过3100W
}

# threshold：判断电器开启的功率阈值（瓦特）
args.threshold = {
    'fridge': 50,       # 冰箱功率>50W认为开启
    'microwave': 200,   # 微波炉功率>200W认为开启
    'kettle': 2000,     # 水壶功率>2000W认为开启
}

# min_on：最小开启时间（时间步数）
args.min_on = {
    'fridge': 10,       # 冰箱至少开启10个时间步
    'microwave': 2,     # 微波炉至少开启2个时间步
    'kettle': 2,        # 水壶至少开启2个时间步
}

# min_off：最小关闭时间（时间步数）
args.min_off = {
    'fridge': 2,        # 冰箱至少关闭2个时间步
    'microwave': 5,     # 微波炉至少关闭5个时间步
    'kettle': 0,        # 水壶可以立即重新开启
}
```

### 3. 训练参数

```python
args.num_epochs = 50        # 训练轮数
args.batch_size = 128       # 批量大小
args.lr = 1e-4              # 学习率
args.mask_prob = 0.25       # BERT掩码概率
```

## 五、数据预处理检查清单

在开始训练前，请检查：

- [ ] CSV文件格式正确（时间戳 + 总功率 + 电器功率）
- [ ] 时间戳格式一致（Unix时间戳或datetime）
- [ ] 时间间隔均匀（如每6秒一个数据点）
- [ ] 功率值为非负数
- [ ] 没有缺失值或已处理缺失值
- [ ] 电器名称与CSV文件中的列名一致
- [ ] config.py中的数据路径设置正确
- [ ] cutoff、threshold等参数根据实际情况调整

## 六、常见问题

### Q1: 如果只有train.csv，没有val.csv和test.csv怎么办？

A: 程序会自动从train.csv中划分一部分数据作为验证集和测试集。建议比例为：
- 训练集：80%
- 验证集：10%
- 测试集：10%

### Q2: 时间戳格式不对怎么办？

A: 程序支持两种格式：
1. Unix时间戳（秒）：如 `1234567890`
2. datetime格式：如 `2024-01-01 00:00:00`

如果格式不匹配，程序会尝试自动转换，失败时会使用索引作为时间。

### Q3: 如何处理缺失值？

A: 程序会自动删除包含缺失值的行。建议在准备数据时：
1. 使用前向填充：`data.fillna(method='ffill')`
2. 或使用插值：`data.interpolate()`

### Q4: 数据量不够怎么办？

A: 可以尝试：
1. 减小window_stride，增加训练样本数量
2. 使用数据增强（如添加噪声）
3. 减小模型复杂度（减少层数）

### Q5: 如何只预测功率，不预测状态？

A: 修改trainer.py中的损失函数，移除margin_loss部分：

```python
# 原代码
total_loss = kl_loss + mse_loss + margin_loss

# 修改后
total_loss = kl_loss + mse_loss
```

## 七、输出结果

训练完成后，结果保存在：

```
experiments/csv_dataset/fridge-microwave-kettle/
├── best_acc_model.pth      # 最佳模型权重
└── test_result.json        # 测试结果
```

test_result.json包含：
- `gt`: 真实功率值
- `pred`: 预测功率值

## 八、下一步

1. 准备你的CSV数据文件
2. 修改config.py设置数据路径
3. 运行训练脚本
4. 查看训练结果和模型性能
5. 根据结果调整参数重新训练
