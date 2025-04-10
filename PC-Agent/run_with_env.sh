#!/bin/bash
# 激活 conda 环境：先加载 conda 脚本，根据 conda 的安装路径进行调整。
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate PCAgent
python -u run.py "$@"
