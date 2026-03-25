from logging import critical

import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch.nn as nn
import numpy as np

# pt自带数据加载
from torch.utils.data import DataLoader, TensorDataset

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


seed = 32
torch.manual_seed(seed)

# 2.前向传播

model = nn.Linear(1,1)  #输入输出维度1

# 1) nn.Sequential pt的模块容器,按顺序组合多个网络层
# forward方法定义模型向前传播逻辑，给定输入经过逻辑得到输出,Seq自己带顺序

'''model = nn.Sequential(nn.Linear(1, 1), nn.Linear(1, 1))'''

# 2) nn.ModuleList([]) 无顺序，需要自己定义forward重写方法自己定义
'''
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel,self).__init__()
        self.layers = nn.ModuleList([nn.Linear(1,1)])
    # 自己写forward顺序
    def forward(self,x):
        for layer in self.layers:
            # 把x丢到模型里面过一遍得到输出
            x = layer(x)
        return x

model = LinearModel()
'''
# 3)nn.ModuleDict({})
#model = nn.ModuleDict({"linear":nn.Linear(1,1)})
'''
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.layers = nn.ModuleDict({"linear": nn.Linear(1, 1)})
    # 自己写forward顺序
    def forward(self,x):
        for layer in self.layers.values():
            # 把x丢到模型里面过一遍得到输出
            x = layer(x)
        return x

model = LinearModel()
'''
# 3.损失函数 和 优化器
criterion = nn.MSELoss() #均方差
optimiser = torch.optim.SGD(model.parameters(), lr=0.5) #梯度下降

# 用于绘制三维图形用
def loss_function(X, Y, w, b):
    predicted_y = np.dot(X, w) + b
    total_loss = np.mean((2 * (predicted_y - Y) ** 2))
    return total_loss

# 用来记录梯度
gd_path = []

# 构建网格点
w_values = np.linspace(-20, 80, 100)
b_values = np.linspace(-20, 80, 100)
W, B = np.meshgrid(w_values, b_values)
loss_values = np.zeros_like(W)

for i, w in enumerate(w_values):
    for j, b in enumerate(b_values):
        loss_values[j, i] = loss_function(x_data, y_data, w, b)

# 创建图形对象和子图布局
fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(2, 2)

# 左上格子
ax2 = fig.add_subplot(gs[0, 0])
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_title("Data")

# 左下格子
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_xlabel("w")
ax3.set_ylabel("b")
ax3.set_title("Contour Plot")

# 整个右侧格子
ax1 = fig.add_subplot(gs[:, 1], projection='3d')
ax1.plot_surface(W, B, loss_values, cmap='viridis', alpha=0.8)
ax1.set_xlabel('w')
ax1.set_ylabel('b')
ax1.set_zlabel('Loss')
ax1.set_title("Surface Plot")

# 获取模型初始化过的参数
w = float(model.weight)
b = float(model.bias)

# 4.迭代过程
epoches = 500

# 在几万个数据的时候需要指定一个batchsize有几个数据输入，此处自动算好样本数，可以设置打乱
# 返回一个可迭代对象，每次迭代为一个batch的数据，为元组包括特征和数据
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)


# 测试形状(squeeze指定位置删除维度,un增加维度)
print(x_train.shape)
print(x_train.unsqueeze(1).shape)

for n in range(1,epoches+1):
    total_loss = 0
    for batch_x, batch_y in dataloader:
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
    w = float(model.weight)
    b = float(model.bias)

        # 显示频率设置
    if n % 10 == 0 or n == 1:
        print(f"epoches:{n}, loss:{avg_loss}")
        # 根据当前参数拟合直线
        x_line = np.linspace(np.min(x_data), np.max(x_data), 100)
        y_line = np.dot(x_line, w) + b

        # 更新子图 1 数据并绘制
        ax2.clear()
        ax2.scatter(x_data, y_data)
        ax2.plot(x_line, y_line, '-')
        ax2.set_title(f"Linear Regression: w={w}, b={b}")
        # 绘制当前w和b的位置
        ax1.scatter(w, b, loss_function(x_data, y_data, w, b), c='black', s=20)

        # 绘制俯视图等高线
        ax3.clear()
        ax3.contourf(W, B, loss_values, levels=20, cmap='viridis')
        ax3.scatter(w, b, c='black', s=20)

        # 绘制梯度下降路径
        if len(gd_path) > 0:
            gd_w, gd_b = zip(*gd_path)
            ax1.plot(gd_w, gd_b,
                     [loss_function(x_data, y_data, np.array(gd_w[i]), np.array(gd_b[i])) for i in
                      range(len(gd_w))],
                     c='black')
            ax3.plot(gd_w, gd_b)
        plt.pause(1)
plt.show()

