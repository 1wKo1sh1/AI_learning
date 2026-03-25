import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import levene
from sipbuild.generator.parser.rules import p_include


# 高维正态分布
def pdf(x, mean, cov, n):
    # 获取均值向量长度，即特征数量
    p1 = 1 / (2 * np.pi) ** (n / 2)
    p2 = 1 / np.sqrt(np.linalg.det(cov))
    ehead = -0.5 * (x - mean).T @ np.linalg.inv(cov) @ (x - mean)
    return p1 * p2 * np.exp(ehead)

# 1.散点输入
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

# 合并数据,创造标签
X = np.concatenate((class1_points, class2_points))
Y = np.concatenate([
    np.zeros(len(class1_points)),
    np.ones(len(class2_points))
    ])

# 2.计算先验
prior_probability = [np.sum(Y == 0) / len(Y), np.sum(Y == 1) / len(Y)]  # 根据标签分别计算元素的先验概率
print(np.array(prior_probability))

# 3.假设数据符合高斯分布(正态分布)
# 类均值
class_means = [np.mean(X[Y == 0]), np.mean(X[Y == 1])]
# 类方差
X1 = X[Y == 0].T
X2 = X[Y == 1].T
class_covs = [np.cov(X1), np.cov(X2)]

# test_point = np.array([3.5, 3])
xspace = np.linspace(0,5, 100)
yspace = np.linspace(0,4, 100)
xs, ys = np.meshgrid(xspace, yspace)
# 预测网格点
test_point = np.c_[xs.ravel(), ys.ravel()]
# 后验
grid_label = []
# 对网格点都进行预测
for point in test_point:
    posterior_probability = []
    for i in range(2):
        # 似然
        n = len(class_means)
        likelihood = pdf(point, class_means[i], class_covs[i], n)
        # 后验概率
        # print(f"属于{i}组数据概率{prior_probability[i] * likelihood:.4f}")
        posterior_probability.append(prior_probability[i] * likelihood)


    pre_class = np.argmax(posterior_probability)
    grid_label.append(pre_class)

# 显示决策边界
# print(f"{test_point}点属于{pre_class}")
grid_label = np.array(grid_label).reshape(xs.shape)

# 绘图
plt.scatter(class1_points[:, 0], class1_points[:, 1], c='b', label='class1--group0')
plt.scatter(class2_points[:, 0], class2_points[:, 1], c='r', label='class2--group1')
# plt.scatter(test_point[0], test_point[1], c='g', label='test_point')

contour = plt.contour(xs, ys, grid_label, levels=[0.5], colors="green")
plt.legend()
# plt.text(test_point[0]+0.1,test_point[1]-0.1, f"class {pre_class}")
plt.show()

