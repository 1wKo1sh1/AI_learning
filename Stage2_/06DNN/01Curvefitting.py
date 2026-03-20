import numpy as np
import matplotlib.pyplot as plt

# 1.数据输入处理
data = np.array([[0.8, 0], [1.1, 0], [1.7, 0], [3.2, 1], [3.7, 1], [4.0, 1], [4.2, 1]])
length = len(data)
x = data[:, 0]
y = data[:, 1]

x_train = np.array(x)
y_train = np.array(y)

# 2.定义神经元、激活函数
def sig(x):
    return 1 / (1 + np.exp(-x))
def d_sig(x):
    return sig(x) * (1 - sig(x))

# 3.迭代
w = 0
b = 0
l = 0.5
epochs = 10000
for epoch in range(1, epochs+1):
    z = w * x_train + b
    # 链式法则 最终z对w偏导和z对b偏导不同
    gradient_w = np.mean(x_train * d_sig(z) * (-2)*(y_train-sig(z)))
    gradient_b = np.mean(1 * d_sig(z) * (-2)*(y_train-sig(z)))
    # 更新
    w = w - l * gradient_w
    b = b - l * gradient_b
    # 损失
    loss = np.mean((y_train-sig(z))**2)
    # 显示频率
    if epoch % 50 == 0 or epoch == 1:
        print(f"epoch: {epoch}, loss: {loss}, w: {w}, b: {b}")