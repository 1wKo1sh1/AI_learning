import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd

# 1数据读取与处理
torch.set_default_device('cuda')

data = pd.read_excel('../dataset/data1.xlsx')

print(data.head(),data.shape)#418*8
# 数据去有序性
'''
觉得并不需要这部分，很明显便利店越多房子越好卖，但是有关于onehot编码的学习意义暂时保留

# 数据处理one-hot（防止12345...导致神经元误认为有序）编码处理，这一列的不同取值更换为二进制列
# 一条列新增好几列(原来数字为0-10，)，然后生成11列通过01编码他们
# 解决了模型认为数字越大结果越大，但是维度变高会产生维度灾难
data_x4 = pd.get_dummies(data, columns=['X4 number of convenience stores'])
print(data_x4.keys())
'''
# 数据划分(时序性直接按比例划分，或者使用sklearn库的划分函数)
x = data.iloc[:,1:7]
y = data.iloc[:,7]
train_size = 0.8
n = int(len(data) * train_size)
x_train, x_test = [np.array(x[:n]),np.array(x[n:])]
y_train, y_test = [np.array(y[:n]),np.array(y[n:])]
# print("y训练\n",y_train.head)
# print(y_test.shape)
# print("y测试\n",y_test.head)
# print(x_train.shape)
# 标准化(去除单位带来的数量级差异，比如1000平方米和1家便利店，1000平方米带来的增益更显著，现在把他们拉到同一比例)
scaler = StandardScaler()
x_train_scaler = scaler.fit_transform(x_train)# 学习参数保存到scaler对象，然后在转换
x_test_scaler = scaler.transform(x_test)#使用学习过的参数进行转换

# pd变为torch识别的张量
x_train_t = torch.tensor(x_train_scaler, dtype=torch.float32)
x_test_t = torch.tensor(x_test_scaler, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 2初始化神经元(前向)
class HousingPricePredictor(nn.Module):
    def __init__(self):
        super(HousingPricePredictor, self).__init__()
        self.layer = nn.Linear(6, 1)

    def forward(self, x):
        x = self.layer(x)
        return x

model = HousingPricePredictor()
# 模型移动到cuda上运行
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
# 3初始化损失函数优化器(反向)
criterion = nn.MSELoss()
# 使用SGD梯度下降(loss=76)，现在更换Adam(loss=76.87)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
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
epochs = 4000
# 定义一个小batch
dataset = TensorDataset(x_train_t, y_train_t)
# 将迭代器放到cuda上
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, generator=torch.Generator(device='cuda'))

for n in range(1,epochs+1):
    total_loss = 0
    for batch_x, batch_y in dataloader:
        # 前向传播
        y_hat = model(batch_x)
        # 计算损失
        loss = criterion(y_hat, batch_y)
        total_loss += loss
        # 清空梯度=>通过损失函数计算目前梯度=>更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 目前(关与batch)平均损失
    avg_loss = total_loss / len(dataloader)
    # 得到目前w和b(方便绘制)
    w = np.array(model.layer.weight.detach().cpu())
    b = np.array(model.layer.bias.detach().cpu())

    # 显示频率
    if n % 100 == 0 or n == 1:
        print(f"epoches:{n}, loss:{avg_loss},\n目前w:{w}\nb:{b}")


# 评估模型
model.eval()
with torch.no_grad():
    # 预测值
    predictions = model(x_test_t)
    # 查看形状发现下文计算出错
    print("预测值形状:",predictions.shape)
    # 均方误差mse
    mse = criterion(predictions, y_test_t)
    # 残差平方和
    ss_res = ((y_test_t - predictions) ** 2).sum()
    # 总平方和
    ss_tor = ((y_test_t - y_test_t.mean()) ** 2).sum()
    # r2
    r2 = 1- (ss_res / ss_tor)
    print(f"MSE: {mse.item():.4f}")
    print(f"R²: {r2.item():.4f}")