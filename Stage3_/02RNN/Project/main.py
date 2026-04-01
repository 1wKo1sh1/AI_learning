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

class PowerData(Dataset):
    def __init__(self, csv_path, input_len):
        # 取值长度(时间序列分析时选取的区间长度，即下文的开始到结束)
        self.input_len = input_len
        self.data = pd.read_csv(csv_path)
        # 关于切出功率，这里不进行过滤，预测实际结果，最终输出时进行过滤

        # 最小值到最大值映射到-1，1进行归一化
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        # 变化形状后应用归一化
        self.data["MinMaxto1"] = self.scaler.fit_transform(self.data["功率(kW)"].values.reshape(-1, 1))

    def __len__(self):
        # 调用len后返回数据长度
        return len(self.data) - self.input_len

    def __getitem__(self, idx):
        # 获得指定索引样本
        start_idx = idx
        end_idx = idx + self.input_len
        feature = self.data["MinMaxto1"].values[start_idx:end_idx]
        target = self.data["MinMaxto1"].values[end_idx:end_idx+1]
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

# 实例化获得数据类
input_len = 6
power_data = PowerData("./A01.csv",input_len)
# 划分数据
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
train_size = int(train_ratio * len(power_data))
val_size = int(val_ratio * len(power_data))
test_size = int(test_ratio * len(power_data))

indices = list(range(len(power_data))) # 得到全长索引列表
#Subset选取大子集的小部分作为新的集合，元素不会被复制，同时有新的内部索引，节省内存
train_data = Subset(power_data, indices[:train_size])
val_data = Subset(power_data, indices[train_size:train_size+val_size])
test_data = Subset(power_data, indices[train_size+val_size:])

# dataloder加载小区块
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)


class DNN(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, output_size=1):
        super().__init__()
        self.aa = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.aa(x)

model = DNN().to(device)
cri= nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) # 加入l2正则化

epochs = 20
for epoch in range(1, epochs+1):
    model.train()
    total_loss = 0
    for batch_feature, batch_target in train_dataloader:
        batch_feature, batch_target = batch_feature.to(device), batch_target.to(device)

        # 前向
        res = model(batch_feature)
        # res形状:[n,1]->[n,]
        # squeeze移除(指定索引)维度大小为1的维度
        # view张量展平,-1代表自动计算对应位置维度
        loss = cri(res.squeeze(1), batch_target.view(-1))
        total_loss += loss

        # 反向
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_dataloader)
    print(f"[{epoch}/{epochs}], Loss {avg_loss:.4f}")