# 导入所需的库
import os
import random

# 导入数据处理和可视化库
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# 导入深度学习框架 PyTorch 相关库
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
import torch.nn.functional as F # 函数调用区别于模块调用，更灵活的测试前向传播结果
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from sklearn.metrics import confusion_matrix
import seaborn as sns


# 设置随机种子以保证结果的可重复性
def setup_seed(seed):
    np.random.seed(seed)  # 设置 Numpy 随机种子
    random.seed(seed)  # 设置 Python 内置随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置 Python 哈希种子
    torch.manual_seed(seed)  # 设置 PyTorch 随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 设置 CUDA 随机种子
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False  # 关闭 cudnn 加速
        torch.backends.cudnn.deterministic = True  # 设置 cudnn 为确定性算法

# 设置随机种子
setup_seed(0)

# 检查是否有可用的 GPU，如果有则使用 GPU，否则使用 CPU
if torch.cuda.is_available():
    device = torch.device("cuda")  # 使用 GPU
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")  # 使用 CPU
    print("CUDA is not available. Using CPU.")

#=========================================正文=======================================

text = "hey how are you"
# 1.数据输入处理
# 数据划分
input_seq = []
output_seq = []
window = 5
for i in range(0, len(text) - window, 1):
    input_seq.append(text[i:i + window])
    output_seq.append(text[i + window])
print("input_seq:", input_seq)
# 去重复
chars = set(text)
chars = sorted(chars)
# print("chars:", chars)
# {" ":0, "a":1 }
char2int = {char: ind for ind, char in enumerate(chars)}
# print("char2int:", char2int)
# {0:" ", 1: "a"}
int2char = dict(enumerate(chars))

# 将字符转成数字编码
input_seq = [[char2int[char] for char in seq] for seq in input_seq]
# print("input_seq:", input_seq)
output_seq = [[char2int[char] for char in seq] for seq in output_seq]

# one-hot 编码
features = np.zeros((len(input_seq), len(chars)), dtype=np.float32)
for i, seq in enumerate(input_seq):
    features[i, seq] = 1.0
input_seq = torch.tensor(features, dtype=torch.float32)
features = np.zeros((len(output_seq), len(chars)), dtype=np.float32)
for i, seq in enumerate(output_seq):
    features[i, seq] = 1.0
output_seq = torch.tensor(features, dtype=torch.float32)

# 2.初始化模型
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = Model(len(chars), 32, len(chars))

cri = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 3.开始迭代
epochs = 1000
for epoch in range(1, epochs + 1):
    output = model(input_seq)
    loss = cri(output, output_seq)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # 8.显示频率设置
    if epoch == 0 or epoch % 50 == 0:
        print(f"Epoch [{epoch}/{epochs}], Loss {loss:.4f}")


# 预测下一个字符
input_text = "ey ho"
# 将字符转成数字编码
input_text = [char2int[char] for char in input_text]
print(input_text)
# one-hot 编码
features = np.zeros((len(chars)), dtype=np.float32)
print(features)
for seq in input_text:
    features[seq] = 1.0
input_text = torch.tensor(features, dtype=torch.float32)
out = model(input_text)
print("next char:", int2char[torch.argmax(out).item()])