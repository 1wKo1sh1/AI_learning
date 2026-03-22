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

# 1.数据输入处理
# 有特殊繁体字符需要big5编码
data = pd.read_csv("../dataset/train.csv", encoding="big5")
print(data.head())
# 前几列不需要
data = data.iloc[:, 3:]
# 缺失的NR元素改为0
data[data=='NR'] = '0'
data = data.astype(float)
# 重新整理为一般的格式
data = data.T
print(data)
# 一共24行、4320列，分离列，每18个为一组分为240组(相当此天的数据且每天24小时)
total_blocks = 240
rows_needed = total_blocks * 24
cols_needed = 18
df = pd.DataFrame(np.nan, index=range(rows_needed), columns=range(cols_needed))
print(df.shape)
# 重新格式化数据
for i in range(0,4320+1-18,18):
    print(i)
    n = i // 18
    df.iloc[24 * n : 24 * (n + 1) , 0 : 18] = data.iloc[0 : 24 , i : 18 + i]
print(df.head())
df.to_csv('测试.csv', index=False, encoding='utf-8-sig')

# 划分
y = df.iloc[9]
X = df.drop(9, axis=1)
print(X.head())
print(y.head())
#==================================================
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
class PM25(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)

model = PM25()
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