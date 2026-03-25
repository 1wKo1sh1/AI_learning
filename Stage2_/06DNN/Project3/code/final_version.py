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
y = df.iloc[:, 9].to_numpy()
X = df.drop(9, axis=1).to_numpy()
#==================================================
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# 标准化
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)# 学习参数保存到scaler对象，然后在转换
x_test_scaled = scaler.transform(x_test)#使用学习过的参数进行转换
# 变为张量
x_train_t = torch.tensor(x_train_scaled, dtype=torch.float32)
x_test_t = torch.tensor(x_test_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

input_size = len(x_train_t[0])
# 2.初始化神经元
class PM25(nn.Module):
    def __init__(self):
        # 调用父类的构造函数
        super().__init__()
        # 输入层 输入特征 输出特征
        self.fc1 = nn.Linear(input_size, 256)
        # 隐藏层 输入特征 输出特征
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        # 输出层 输入特征 输出特征
        self.fc4 = nn.Linear(256, 1)
        # 定义激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # 输入数据经过第一个Linear和ReLU激活
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

model = PM25()
model.to(device)
# 损失函数与优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 3.迭代过程
# 初始化随机种子
seed = 32
torch.manual_seed(seed)
# 定义一个小batch
dataset = TensorDataset(x_train_t, y_train_t)
# 将迭代器放到cuda上
dataloader = DataLoader(dataset, batch_size=100, shuffle=True, generator=torch.Generator(device='cuda'))
# 迭代次数
epochs = 1000
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
    # w = model.layers[0].weight.detach().cpu().numpy()
    # b = model.layers[0].bias.detach().cpu().numpy()

    # 显示频率
    if epoch % 50 == 0 or epoch == 1:
        print(f"epoches:[{epoch}], loss:[{avg_loss:.4f}]")

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

    # 实际值和预测值关系
    plt.figure(1,figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([0, 100], [0, 100], 'r--', label='T=P')  # 对角线
    plt.xlabel('T')
    plt.ylabel('P')
    plt.title('True vs Predicted')
    plt.legend()
    plt.grid(True)

    # 每个测试轮数的实际值和预测值
    plt.figure(2,figsize=(6, 6))
    plt.plot(y_true[-100:], label='T', marker='o')
    plt.plot(y_pred[-100:], label='P', marker='*')

    plt.xlabel('epoch')
    plt.ylabel('P Or T')
    plt.title('P Or T On epoch')
    plt.legend()
    plt.grid(True)
    plt.show()