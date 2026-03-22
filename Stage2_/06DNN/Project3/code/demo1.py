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