from logging import critical

import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch.nn as nn
import numpy as np

# pt自带数据加载
from torch.utils.data import DataLoader, TensorDataset

from torchsummary import summary



class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel,self).__init__()
        self.layers = nn.ModuleList([nn.Linear(1,1),nn.Linear(1,1)])
    # 自己写forward顺序
    def forward(self,x):
        for layer in self.layers:
            # 把x丢到模型里面过一遍得到输出
            x = layer(x)
        return x

# 1.输入数据
data = [[-0.5, 7.7], [1.8, 98.5], [0.9, 57.8], [0.4, 39.2], [-1.4, -15.7], [-1.4, -37.3], [-1.8, -49.1], [1.5, 75.6], [0.4, 34.0], [0.8, 62.3]]
data = np.array(data)
x_data = data[:, 0]
y_data = data[:, 1]

x_train = torch.tensor(x_data,dtype=torch.float32)
y_train = torch.tensor(y_data,dtype=torch.float32)

# 封装张亮，特征和标签组成数据集
# 返回按照索引获得特征和标签
dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

seed = 32
torch.manual_seed(seed)

model = nn.Linear(1,1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 3.损失函数 和 优化器
criterion = nn.MSELoss() #均方差
optimiser = torch.optim.SGD(model.parameters(), lr=0.05) #梯度下降

# 获取模型初始化过的参数
w = float(model.weight)
b = float(model.bias)

# 4.迭代过程
epoches = 500

# 在几万个数据的时候需要指定一个batchsize有几个数据输入，此处自动算好样本数，可以设置打乱
# 返回一个可迭代对象，每次迭代为一个batch的数据，为元组包括特征和数据



# 测试形状(squeeze指定位置删除维度,un增加维度)
print(x_train.shape)
print(x_train.unsqueeze(1).shape)

gd_path = []
for n in range(1,epoches+1):
    total_loss = 0
    for batch_x, batch_y in dataloader:
        # 数据转移到gpu
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        # 记录梯度
        gd_path.append((w,b))
        # 前向传播
        y_hat = model(batch_x.unsqueeze(1))
        # 计算损失
        loss = criterion(y_hat.squeeze(1), batch_y)
        total_loss += loss
        # 清空之前存储在优化器里面的梯度 (计算一次内部存储了现在的梯度，需要清空)
        optimiser.zero_grad()
        # 计算损失函数关于模型参数的梯度
        loss.backward()
        # 根据优化算法更新参数
        optimiser.step()

    # 平均损失
    avg_loss = total_loss / len(dataloader)
    # 当前参数
    w = model.weight
    b = model.bias



# 直接输出模型(不能检测模型是否正确)
print(model)
# 1.summary打印
# print(next(model.parameters()).device)
# print(summary(model, (1,), device='cpu'))
print(summary(model, (1,), device='cuda'))
# 2.netron(网页,本地)，可能出问题，保存时
# torch.jit.script(model)
# 使用torch.jit.save(script_model,"script_model.pth")
# 保存脚本文件修复不连线问题
torch.save(model, './netron_test.pth')