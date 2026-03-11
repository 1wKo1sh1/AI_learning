"""
二分类 => 多分类
需要变化bayes公式

"""
import numpy as np
import matplotlib.pyplot as plt


def pdf(x, mean, cov):
    # 获取均值向量的长度，即特征的数量
    n = len(mean)
    # 计算PDF的系数部分
    coeff = 1 / ((2 * np.pi) ** (n / 2) * np.sqrt(np.linalg.det(cov)))
    # 计算PDF的指数部分
    exponet = -0.5 * np.dot(np.dot((x - mean).T, np.linalg.inv(cov)), (x - mean))
    return coeff * np.exp(exponet)


# 1.散点输入
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

class3_points = np.array([[3.3, 1.2],
                          [3.8, 0.9],
                          [3.3, 0.6],
                          [2.8, 1.3],
                          [3.5, 0.6],
                          [4.2, 0.3],
                          [3.1, 0.9]])
# 合并数据集、创造标签
X = np.concatenate((class1_points, class2_points, class3_points))
y = np.concatenate((np.zeros(len(class1_points)), np.ones(len(class2_points)), np.ones(len(class3_points)) * 2))

# 2. 计算先验概率(每一个类别的数据在数据集中的比例)
prior_probability = [np.sum(y == 0) / len(y),
                       np.sum(y == 1) / len(y),
                       np.sum(y== 2) / len(y)]
print(prior_probability)

# 3.计算高斯分布的概率密度函数
# 求解每个类别的均值
class_means = [np.mean(X[y == 0], axis=0),
               np.mean(X[y == 1], axis=0),
               np.mean(X[y == 2], axis=0)]
# print(class_means)

# 求解每个类别的协方差矩阵
X_y_0 = X[y == 0].T
X_y_1 = X[y == 1].T
X_y_2 = X[y == 2].T
class_covs = [np.cov(X_y_0), np.cov(X_y_1), np.cov(X_y_2)]

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
    for i in range(len(class_means)):
        # 似然
        likelihood = pdf(point, class_means[i], class_covs[i])
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
plt.scatter(class3_points[:, 0], class3_points[:, 1], c='c', label='class3--group2')
# plt.scatter(test_point[0], test_point[1], c='g', label='test_point')

contour = plt.contour(xs, ys, grid_label, colors="green")
plt.legend()
# plt.text(test_point[0]+0.1,test_point[1]-0.1, f"class {pre_class}")
plt.show()


