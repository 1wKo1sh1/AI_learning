import numpy as np
from matplotlib import pyplot as plt

# 1.数据集输入
data = [[0.8, 1.0], [1.7, 0.9], [2.7, 2.4], [3.2, 2.9], [3.7, 2.8], [4.2, 3.8], [4.2, 2.7]]
data = np.array(data)
# 特征标签分离(x为特征 y为标签 即x的不同引起y不同)
x_data = data[:, 0]
y_data = data[:, 1]

# 2.向前传播
# 实例化y=wx+b
w = 0
w_origin = w
b = 0
y_hat = w * x_data + b
learning_rate = 0.01

# 3.单点误差
e = y_data - y_hat
print(f"单点误差:{e}")

# 4.均方误差(损失函数)
e_bar = np.mean((y_data - y_hat) ** 2)
print(f"均方误差:{e_bar}")

# 定义图
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

step = 10
for i in range(step):
    # 清空画布
    ax1.cla()
    ax2.cla()
    # e对w求导
    gradient = 2 * w_origin * np.mean(x_data ** 2) - 2 * np.mean(x_data * y_data)
    # 反向传播(更新)
    w_new = w_origin - learning_rate * gradient
    # w_new处损失
    y_hat_new = w_new * x_data + b
    e_bar_new = np.mean(y_hat_new - y_data) ** 2
    # 设置坐标轴
    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, 6)
    ax1.set_xlabel("x-----")
    ax1.set_ylabel("y-----")
    # 绘制散点
    ax1.scatter(x_data, y_data, color="blue")
    # 绘制拟合
    ym , YM = w_new*0 + b , w_new*5 + b
    ax1.plot([0,5],[ym,YM],color="red",linewidth=1)
    # 点到拟合的连接线(y轴)
    for x, y_true, y_pre in zip(x_data, y_data, y_hat_new):
        ax1.plot([x,x],[y_true,y_pre],color="green",linestyle="--",linewidth=2)

    # 绘制图2
    w_values = np.linspace(0,3,100)
    # 对w求对应e值(e_bar = np.mean((y_data - y_hat) ** 2))
    # 此把yhat换成w*xdata+b,换为计算值因为广播不了因为大小比较复杂
    e_values = [np.mean((y_data - (w_value * x_data + b)) ** 2) for w_value in w_values]
    # 绘制均方差
    ax2.plot(w_values, e_values, color="green", linewidth=3)
    ax2.plot(w_new, e_bar_new, marker="o", color="red")
    plt.pause(0.5)
    w_origin = w_new
plt.show()


