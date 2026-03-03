'''
KNN(k临近法)是什么 => 定义:基于实例的学习,通过测量不同样本之间的距离,进行分类和回归
                    原理:基于实例属于懒惰学习,没有显式的学习过程,没有训练阶段,通过测量距离进行分类和回归
                    特点:算法简单易懂,易于实现;无训练阶段直接分类回归;适用于多分类问题;对数据维度和大小不敏感

最简单的思想:寻找最近的k个数据,对预测数据进行投票,票数最高的标签作为预测数据标签(提高泛化能力),k=1时变成近邻算法

结论:
    k太小 => 优点:复杂数据集,小k更详细描述决策边界,模型更灵活
            缺点:容易受到局部结构的影响,模型对噪声和异常值的影响更大
    k太大 => 优点:考虑全局信息,对于平湖数据集较大k提供稳定边界
            缺点:复杂数据集,大k导致模型简单,无法捕捉局部特征
    k叫什么 => 需要人为确定参数叫 超参数(hyperparameter)

验证k值:
    准备数据集 => k交叉验证:确定k折(例如5,7) => 划分数据集:为k个大小相似子集(提升泛化能力) => 循环训练评估:k-1个子集作为训练,1个测试 =>
    计算平均性能:k次验证的性能取平均作为评估 => 选择最佳k值:选择交叉验证中准确率最高的k

曼哈顿L1/欧氏L2
'''
# 手写实现
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

# 1.定义数据集
point1 = [[7.7, 6.1], [3.1, 5.9], [8.6, 8.8], [9.5, 7.3], [3.9, 7.4], [5.0, 5.3], [1.0, 7.3]]
point2 = [[0.2, 2.2], [4.5, 4.1], [0.5, 1.1], [2.7, 3.0], [4.7, 0.2], [2.9, 3.3], [7.3, 7.9]]
point3 = [[9.2, 0.7], [9.2, 2.1], [7.3, 4.5], [8.9, 2.9], [9.5, 3.7], [7.7, 3.7], [9.4, 2.4]]
# 合并(axis为展平垂直合并,1为按照矩阵垂直合并)
train_data = np.concatenate((point1, point2, point3), axis=0)
print(train_data)
# 创建标签 (7个0 7个1 7个2)
train_label = np.array([0]*len(point1) + [1]*len(point2) + [2]*len(point3))

# 2.构建KNN算法，实例化KNN算法，KNN训练(懒惰算法,此处训练即赋值操作)
# 初始化k近邻分类器(此处控制k的大小来适应不同情况)
knn_clf = KNeighborsClassifier(1)
# 训练(fit代表内部的赋值过程)
knn_clf.fit(train_data, train_label)

# 3.决策边界,设定未知点
axis = [0, 10, 0, 10]
# 生成坐标网络
x0, x1 = np.meshgrid(
    np.linspace(axis[0], axis[1], 100), # x上均匀点
    np.linspace(axis[2], axis[3], 100) # y上均匀点
)
# 展平后合并(竖向排开一堆点)
axis_xy = np.c_[x0.ravel(), x1.ravel()] # [0,0],[0,1],...,[10,10]一共10000个网格点

# 4.KNN预测与绘制决策边界
# 对全部点进行预测
y_perdict = knn_clf.predict(axis_xy) # 目的：得到整个区域每个位置的类别标签
y_perdict = y_perdict.reshape(x0.shape)# [0, 0, 0, ..., 1, 1, 1, ..., 2, 2, 2]
# 等高线绘制(将三个标签当成3种高度)
# X,Y,Z(将高度作为标签绘制等高线)
plt.contour(x0, x1,y_perdict)
# 绘制散点
plt.scatter(train_data[train_label == 0, 0], train_data[train_label == 0, 1],marker='^')
plt.scatter(train_data[train_label == 1, 0], train_data[train_label == 1, 1],marker='*')
plt.scatter(train_data[train_label == 2, 0], train_data[train_label == 2, 1],marker='s')
plt.show()

