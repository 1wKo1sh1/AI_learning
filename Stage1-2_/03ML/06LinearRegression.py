import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 1、散点输入
data = [[-0.5, 7.7], [1.8, 98.5], [0.9, 57.8], [0.4, 39.2], [-1.4, -15.7],
        [-1.4, -37.3], [-1.8, -49.1], [1.5, 75.6], [0.4, 34.0], [0.8, 62.3]]
# 转换为 NumPy 数组
data = np.array(data)
# 提取 x_data 和 y_data
x_data = data[:, 0]
y_data = data[:, 1]

# 2.初始化参数
# 一般参数
w = 0
b = 0
# 超参数
learning_rate = 0.01

# 3.求损失函数,实际代码反向传播没有用到,但是绘制图像需要
def loss_func(X,Y,w,B):
    _y_hat = X * w + B
    loss = np.mean(( _y_hat - Y ) **2 )
    return loss

# 定义图对象
fig = plt.figure(figsize=(12, 6))
# 划分两行两列布局对象
gs = gridspec.GridSpec(2, 2)
# 左上角区块
ax2 = fig.add_subplot(gs[0, 0])
ax2.set_xlabel("x")
ax2.set_ylabel("y")
# 左下区块
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_xlabel("w")
ax3.set_ylabel("b")
# 右边区块projection指定为3d对象
ax1 = fig.add_subplot(gs[:, 1], projection="3d")
# 绘制3d网格
w_values = np.linspace(-20, 80, 100)
b_values = np.linspace(-20, 80, 100)
W, B = np.meshgrid(w_values, b_values)
loss_values = np.zeros_like(W)
# 计算损失值

for i,w in enumerate(w_values):
    for j,b in enumerate(b_values):
        loss_values[j,i] = loss_func(x_data,y_data, w, b)

ax1.plot_surface(W, B, loss_values,alpha=0.6,cmap="viridis")
ax1.set_xlabel("w")
ax1.set_ylabel("b")
ax1.set_zlabel("loss")

# 4.开始迭代
num_iterations = 500
path = [] # 历史路径
for n in range(1, num_iterations + 1):
    path.append([w, b,loss_func(x_data,y_data,w,b)])
    # 5.反向传播
    y_hat = w * x_data + b
    gradient_w = np.mean(2 * (y_hat - y_data) * x_data)
    gradient_b = np.mean(2 * (y_hat - y_data))
    # 更新参数
    w = w - learning_rate * gradient_w
    b = b - learning_rate * gradient_b

    # 6.显示频率设置
    if n % 10 == 0 or n == 1 or n == num_iterations:
        print(f"当前w: {w}, b: {b}")
        # 更新ax2
        ax2.clear()
        xm = x_data.min()
        XM = x_data.max()
        ym = w * xm + b
        YM = w * XM + b
        ax2.scatter(x_data, y_data)
        ax2.plot([xm, XM], [ym, YM], c="r")

        # 更新ax3
        ax3.contourf(W, B, loss_values,levels=20)
        ax3.scatter(w , b, c="r")
        # 绘制拖尾
        if len(path) > 0:
            path_w,path_b,path_loss = zip(*path)
            ax3.plot(path_w, path_b, c="b")
            ax1.plot(path_w, path_b,path_loss, c="b")
        # 更新ax1
        ax1.scatter(w,b,loss_func(x_data,y_data,w,b),c="r")


        plt.pause(1)

# 7.绘图
