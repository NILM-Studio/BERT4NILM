#!/bin/bash
#SBATCH --job-name=bert4nilm
#SBATCH --output=bert4nilm_%j.out
#SBATCH --error=bert4nilm_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# 激活 conda 环境
module purge
module load cuda/12.1
module load python/3.10
source /path/to/conda/etc/profile.d/conda.sh
conda activate bert4nilm

# 设置工作目录
cd $SLURM_SUBMIT_DIR

# 运行训练
python train.py --config config.json

# 或者使用其他配置文件
# python train.py --config config_uk_dale.json
# python train.py --config config_redd.json
