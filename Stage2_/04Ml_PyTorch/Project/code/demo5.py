import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pandas as pd

# 1数据读取与处理
torch.set_default_device('cuda')

data = pd.read_excel('../dataset/data1.xlsx')

print(data.head(),data.shape)#418*8

# 数据处理one-hot（防止12345...导致神经元误认为有序）编码处理，这一列的不同取值更换为二进制列
# 一条列新增好几列(原来数字为0-10，)，然后生成11列通过01编码他们
# 解决了模型认为数字越大结果越大，但是维度变高会产生维度灾难
data = pd.get_dummies(data, columns=['X4 number of convenience stores'])
# print(data_x4.keys())

# 数据划分(时序性直接按比例划分，或者使用sklearn库的划分函数)
feature_columns = [
    'X1 transaction date',
    'X2 house age',
    'X3 distance to the nearest MRT station',
    'X5 latitude',
    'X6 longitude',
    'X4 number of convenience stores_0',
    'X4 number of convenience stores_1',
    'X4 number of convenience stores_2',
    'X4 number of convenience stores_3',
    'X4 number of convenience stores_4',
    'X4 number of convenience stores_5',
    'X4 number of convenience stores_6',
    'X4 number of convenience stores_7',
    'X4 number of convenience stores_8',
    'X4 number of convenience stores_9',
    'X4 number of convenience stores_10'
]
x = data[feature_columns]
y = data['Y house price of unit area']
# train_size = 0.8
# n = int(len(data) * train_size)
train_ratio = 0.8
x, y = shuffle(X, y)
x_train = x[:int(train_ratio * len(data))]
x_test = x[int((train_ratio) * len(data)):]
y_train = y[:int(train_ratio * len(data))]
y_test = y[int((train_ratio) * len(data)):]
# 标准化(去除单位带来的数量级差异，比如1000平方米和1家便利店，1000平方米带来的增益更显著，现在把他们拉到同一比例)
scaler = StandardScaler()
x_train_scaler = scaler.fit_transform(x_train)# 学习参数保存到scaler对象，然后在转换
x_test_scaler = scaler.transform(x_test)#使用学习过的参数进行转换

# pd变为torch识别的张量
x_train_t = torch.tensor(x_train_scaler, dtype=torch.float32)
x_test_t = torch.tensor(x_test_scaler, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
y_test_t = torch.tensor(y_test.values, dtype=torch.float32)

# 2初始化神经元(前向)
class HousingPricePredictor(nn.Module):
    def __init__(self):
        super(HousingPricePredictor, self).__init__()
        self.layer = nn.Linear(16, 1)

    def forward(self, x):
        x = self.layer(x)
        return x

model = HousingPricePredictor()
# 模型移动到cuda上运行
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
# 3初始化损失函数优化器(反向)
criterion = nn.MSELoss()
# 使用SGD梯度下降(loss=188.09)优化器发现效果不好，现在更换Adam(loss=198.13)发现更不好
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-5)
# 4初始化可视化

# 5实际迭代过程
# 初始化种子
seed = 32
torch.manual_seed(seed)
# 初始化cuda全局种子
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
# 梯度
w = model.layer.weight.detach()
b = model.layer.bias.detach()
# 迭代次数
epochs = 5000
for n in range(1,epochs+1):
    total_loss = 0
    # 前向传播
    y_hat = model(x_train_t)
    # 计算损失
    loss = criterion(y_hat, y_train_t)
    # 清空梯度=>通过损失函数计算目前梯度=>更新参数
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 得到目前w和b(方便绘制)
    w = np.array(model.layer.weight.detach().cpu())
    b = np.array(model.layer.bias.detach().cpu())

    # 显示频率
    if n % 50 == 0 or n == 1:
        print(f"epoches:{n}, loss:{loss},\n目前w:{w}\nb:{b}")