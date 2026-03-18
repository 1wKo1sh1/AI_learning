import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

# 定义默认为gpu
torch.set_default_device('cuda')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1.数据输入处理(使用pyarrow引擎加速读取速度但是格式会变为独特格式后面转换出问题)
dataset = pd.read_csv('../dataset/agaricus-lepiota.data', header=None, engine='c')
dataset[0] = dataset[0].map({'e': True, 'p': False})
print(dataset.head())
dataset = pd.get_dummies(dataset,columns=list(range(1,23)))
print(dataset.head()) # 8xxx x 118
# 获取特征与分类
X = dataset.iloc[:,1:].to_numpy()
y = dataset.iloc[:,0].to_numpy()
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# 标准化(原文为01已经不需要标准化)
x_train_scaled = x_train
x_test_scaled = x_test
# 变为张量
x_train_t = torch.tensor(x_train_scaled, dtype=torch.float32)
x_test_t = torch.tensor(x_test_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# 2.初始化神经元
input_size = len(x_train_t[0])
class MushroomsEdible(nn.Module):
    def __init__(self):
        super(MushroomsEdible, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)

model = MushroomsEdible()
model.to(device)
# 损失函数与优化器
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# 3.迭代过程
# 初始化随机种子
seed = 32
torch.manual_seed(seed)
# 定义一个小batch
dataset = TensorDataset(x_train_t, y_train_t)
# 将迭代器放到cuda上
dataloader = DataLoader(dataset, batch_size=500, shuffle=True, generator=torch.Generator(device='cuda'))
# 迭代次数
epochs = 100
for epoch in range(1,1+epochs):
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

    # 计算平均损失
    avg_loss = total_loss / len(dataloader)
    # 得到目前w和b(方便绘制)
    w = model.layers[0].weight.detach().cpu().numpy()
    b = model.layers[0].bias.detach().cpu().numpy()

    # 显示频率
    if epoch % 50 == 0 or epoch == 1:
        print(f"epoches:{epoch}, loss:{avg_loss},\n目前w:{w}\nb:{b}")


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

    y_pred = predictions.cpu().numpy()
    y_true = y_test_t.cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Prediction')  # 对角线
    plt.xlabel('True')
    plt.ylabel('Pred')
    plt.title('True vs Predicted')
    plt.legend()
    plt.grid(True)
    plt.show()
