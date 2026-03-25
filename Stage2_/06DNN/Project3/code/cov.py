
import pandas as pd
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader



# 设置随机种子保证结果的可重复性
def setup_seed(seed):
    # 设置 Numpy 随机数种子，确保Numpy生成的随机数序列一致
    np.random.seed(seed)

    # 设置Python内置随机数种子，保证Python内置的随机函数生成的随机数一致
    random.seed(seed)

    # 设置Python哈希种子，避免不同运行环境下哈希结果不同，影响随机数生成
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 设置PyTorch 随机种子，使PyTorch生成的随机数序列可以重复
    torch.manual_seed(seed)

    # 检查是否有可用的CUDA设备（GPU）
    if torch.cuda.is_available():
        # 设置 CUDA 随机种子，保证在GPU上的随即操作可重复
        torch.cuda.manual_seed(seed)
        # 为所有 GPU 设置随机种子
        torch.cuda.manual_seed_all(seed)
        # 关闭 cudnn 自动寻找最优算法加速的功能，保证结果可重复
        torch.backends.cudnn.benchmark = False
        # 设置 cudnn 为确定性算法，确保每次运行结果一致
        torch.backends.cudnn.deterministic = True

if torch.cuda.is_available():
    device = 'cuda'
    print('CUDA is useful!!')
else:
    device = 'cpu'
    print('CUDA is not useful!!')

setup_seed(0)

# 设置 pandas 显示选项，以便显示更多的列和行的内容
# 最多显示1000列
pd.set_option('display.max_columns', 1000)
# 显示宽度为1000
pd.set_option('display.width', 1000)
# 每列最多显示1000个字符
pd.set_option('display.max_colwidth', 1000)


# 读取数据集，需要注意的是；编码格式必须为 big5
train_data = pd.read_csv('../dataset/train.csv', encoding='big5')
# 查看读取进来的前5行数据，确保数据被正确读取
# print(train_data.head())

# 打印数据集的信息，查看数据集的情况
# print(train_data.info())


# 选取从第3列开始到最后的所有列作为特征数据
train_data = train_data.iloc[:, 3:]

# 将数组中值为NR的元素替换为0
train_data[train_data == 'NR'] = '0'
train_data = train_data.astype(float)

# 将train_data转换为Numpy数组
numpy_data = train_data.to_numpy()

# # 检查数据集中缺失值情况
# print(train_data.isnull().sum())

# 创建一个列表，用来存储拆分后的数据
datas = []

# 按照 步长为18 分割数据
for i in range(0, 4320, 18):
    datas.append(numpy_data[i:i+18, :])

# print(datas)

# 将datas 转换为Numpy数组
datas_array = np.array(datas, dtype=float)

# print(datas_array.shape)

# 对数据进行维度变换和重塑，转为DataFrame格式
train_data = pd.DataFrame(datas_array.transpose(1, 0, 2).reshape(18, -1).T)

# print(test.shape)

# 计算特征相关性矩阵
corr = train_data.corr()

# 绘制相关性热图
plt.figure(0)

# seaborn库：基于matplotlib的高级可视化库
# heatmap()：热力图，用于展示矩阵数据，通过颜色的深浅表示数值大小，常用于展示相关性矩阵、混淆矩阵等内容，可以帮助用户快速
# 发现数据之间的关系
sns.heatmap(corr, annot=True)


# 从相关性矩阵里筛选比较重要的特征
important_features = []
for i in range(len(corr.columns)):
    # 选择与第9列相关性系数绝对值大于0.2的特征
    if abs(corr.iloc[i, 9]) > 0.2:
        important_features.append(corr.columns[i])

plt.show()