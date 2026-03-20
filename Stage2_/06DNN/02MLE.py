'''
MSE利用距离测度衡量关系
MLE利用概率测度衡量关系
'''
import numpy as np
import matplotlib.pyplot as plt

# 1.数据输入处理
class1_points = np.array([[1.9, 1.2],
                          [1.5, 2.1],
                          [1.9, 0.5],
                          [1.5, 0.9],
                          [0.9, 1.2],
                          [1.1, 1.7],
                          [1.4, 1.1]])

class2_points = np.array([[3.2, 3.2],
                          [3.7, 2.9],
                          [3.2, 2.6],
                          [1.7, 3.3],
                          [3.4, 2.6],
                          [4.1, 2.3],
                          [3.0, 2.9]])
# 横坐标作为特征1，纵坐标作为特征2，两类标签
X = np.concatenate((class1_points, class2_points), axis=0) # 竖向拼接
label = np.concatenate((np.zeros(len(class1_points)), np.ones(len(class2_points)))).reshape(-1,1)

# 2.定义神经元
# 损失
def get_loss(a):
    loss = -np.mean(label * np.log(a) + (1 - label) * np.log(1 - a))
    return loss
def derror(a):
    return (a - label) / (a * (1 - a))

# 激活函数
def sig(z):
    return 1 / (1 + np.exp(-z))
def d_sig(z):
    return sig(z) * (1 - sig(z))

# 前向传播
def forward(W ,b):
    Z = X @ W + b
    return Z
def dz_W():
    return X
def dz_b():
    return 1
# 绘图类
class Plotting():
    def __init__(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1)
        self.epoch_list = []
        self.loss_list = []
    # 绘制类别1和类别2的三点，并且画出决策边界
    def ax1_plot(self):
        self.ax1.clear()
        self.ax1.scatter(X[:, 0][:len(class1_points)], X[:, 1][:len(class2_points)])
        self.ax1.scatter(X[:, 0][len(class1_points):], X[:, 1][len(class2_points):])
        self.ax1.plot((x1_min, x1_max), (x2_min, x2_max))
    # 绘制损失与迭代次数关系
    def ax2_plot(self):
        self.ax2.clear()
        self.epoch_list.append(epoch)
        self.loss_list.append(loss)
        self.ax2.plot(self.epoch_list, self.loss_list)
        plt.pause(1)
    def pltshow(self):
        plt.show()
# 3.迭代
W = np.array([0,0])
b = 0
l = 0.1
plotting = Plotting()
epochs = 1000
for epoch in range(1,epochs+1):
    Z = forward(W, b).reshape(-1,1)
    # print("z:",Z,Z.shape)
    a = np.array(sig(Z)).reshape(-1,1)
    # print("a:",a,a.shape)
    gradient_W = np.mean(dz_W() * d_sig(Z) * derror(a),axis=0)
    # print("gradient_W:",gradient_W)
    gradient_b = np.mean(dz_b() * d_sig(Z) * derror(a))
    # print("gradient_b:", gradient_b)
    W = W - l * gradient_W
    b = b - l * gradient_b

    # 显示
    if epoch % 50 == 0 or epoch == 1:
        loss = get_loss(a)
        print(f"epoch: {epoch}, loss: {loss}, W: {W}, b: {b}")
        x1_min, x1_max = X[:,0].min(), X[:,0].max()
        x2_min, x2_max = -(W[0] * x1_min + b) / W[1], -(W[0] * x1_max + b) / W[1]

        plotting.ax1_plot()
        plotting.ax2_plot()

plotting.pltshow()