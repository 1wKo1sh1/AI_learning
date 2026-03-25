import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
# 定义默认为gpu
torch.set_default_device('cuda')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1.数据读取处理
data = pd.read_csv('../dataset/abalone.data')
print(data.head())
data.columns = ['sex', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
data = pd.get_dummies(data,columns=['sex'])
print(data.head())
X = data[['a','b','c','d','e','f','g','sex_F','sex_I','sex_M']]
y = data['h']
print(X.head())
# 划分训练集
train_size = 0.8
n = int(len(data) * train_size)
x_train, x_test = [np.array(X[:n]),np.array(X[n:])]
y_train, y_test = [np.array(y[:n]),np.array(y[n:])]
# 标准化
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
# 变为张量
x_train_t = torch.tensor(x_train_scaled, dtype=torch.float32)
x_test_t = torch.tensor(x_test_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# 2.初始化模型
n = len(X.columns)
class AbaloneRing(nn.Module):
    def __init__(self, n):
        super(AbaloneRing, self).__init__()
        self.linear1 = nn.Linear(n, 1)

    def forward(self, x):
        x = self.linear1(x)
        return x

print("当前自变量:",n)
model = AbaloneRing(n)
# 转移模型到gpu
model = model.to(device)

# 3.损失函数优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# 4.可视化(可选)

# 5.迭代过程
# 种子
seed = 514
torch.manual_seed(seed)
# 目前权重与偏置
w, b = model.linear1.weight.detach(), model.linear1.bias.detach()

# 迭代次数
epochs = 1000
# 数据转移到gpu
x_train_t, y_train_t = x_train_t.to(device), y_train_t.to(device)

for n in range(1,epochs+1):
    # 前向
    y_hat = model(x_train_t)
    # 损失
    loss = criterion(y_hat, y_train_t)
    # 清空梯度=>通过损失函数计算目前梯度=>更新参数
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 显示频率
    if n % 100 == 0 or n == 1:
        print(f'===============第{n}次===========================')
        print(f"epoches:{n}, loss:{loss},\n目前w:{w}\nb:{b}")


# 评估模型
model.eval()
with torch.no_grad():
    # 预测值
    predictions = model(x_test_t)
    # 查看形状防止下文计算出错
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

# 绘制真实值与预测值对比
predictions = predictions.cpu().numpy()
y_test_t = y_test_t.cpu().numpy()

plt.figure(figsize=(6,6))
plt.scatter(y_test_t, predictions)
xx = np.linspace(min(y_test_t), max(y_test_t), 100)
plt.plot(xx, xx,c='r')

plt.show()