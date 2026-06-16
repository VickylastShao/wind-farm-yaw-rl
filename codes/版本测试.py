# check_env.py
import sys
import numpy
import torch
import stable_baselines3

print("\n--- Python 环境诊断报告 ---")
print(f"🐍 Python 解释器路径: {sys.executable}")
print("-" * 25)
print(f" Numpy 版本: {numpy.__version__}")
print(f" PyTorch 版本: {torch.__version__}")
print(f" Stable-Baselines3 版本: {stable_baselines3.__version__}")
print("--------------------------\n")