import torch
print(torch.__version__)  # 检查 PyTorch 版本
print(torch.backends.mps.is_available())  # 是否支持 MPS
print(torch.backends.mps.is_built())  # PyTorch 是否编译了 MPS 支持