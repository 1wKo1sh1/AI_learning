'''
准备数据:准备好特征标签数据集和预测点数据 => 求距离:求训练点到预测点的l2 => 距离排序:排序得到索引值
=> 获得k个最近的距离:取出k个距离最小的索引(最近的k个元素索引) => 结果
'''
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


# 1.定义数据集和测试点
# 三组点
point1 = [[7.7, 6.1], [3.1, 5.9], [8.6, 8.8], [9.5, 7.3], [3.9, 7.4], [5.0, 5.3], [1.0, 7.3]]
point2 = [[0.2, 2.2], [4.5, 4.1], [0.5, 1.1], [2.7, 3.0], [4.7, 0.2], [2.9, 3.3], [7.3, 7.9]]
point3 = [[9.2, 0.7], [9.2, 2.1], [7.3, 4.5], [8.9, 2.9], [9.5, 3.7], [7.7, 3.7], [9.4, 2.4]]
# 合并
train_data = np.concatenate((point1,point2,point3))
# 分类(加标签)
train_label = np.array([0]*len(point1) + [1]*len(point2) + [2]*len(point3))
# 定义预测点坐标
predict_point = np.array([10, 4.2])
# 2.定义k值
K = 3

# 3.求距离,获得前k个最短距离,最短距离对应点的坐标
# 使用numpy广播机制求距离(动帮你“拉伸”小数组，让它能和大数组做运算，而不用手动复制数据。)
# axis=1 代表求和时堆列对每一列单独求和
distance = np.sqrt(np.sum((predict_point - train_data) ** 2, axis=1))
# 排序[1,5,4]=>[0,2,1]即排序之后的索引
index = np.argsort(distance)
# 前k个索引
nearest_index = index[:K]
nearest_point = []
nearest_distance = []
nearest_label = []
# 根据索引获得信息
for index in nearest_index:
    nearest_point.append(train_data[index])
    nearest_distance.append(distance[index])
    # 拿到最近元素的标签,查看属于哪类
    nearest_label.append(train_label[index])

counter = Counter(nearest_label)
print("数据点label为:", counter.most_common()[0][0])
# 绘图
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(train_data[train_label == 0,0], train_data[train_label == 0,1],marker='*')
plt.scatter(train_data[train_label == 1,0], train_data[train_label == 1,1],marker='^')
plt.scatter(train_data[train_label == 2,0], train_data[train_label == 2,1],marker='s')

plt.scatter(predict_point[0], predict_point[1], marker='o')

for i in range(K):
    # 划线([两点横坐标],[两点纵坐标])
    plt.plot([predict_point[0], nearest_point[i][0]], [predict_point[1], nearest_point[i][1]])
    # 线上标签 (文字,位置)
    plt.annotate(f"{nearest_distance[i]:2.2f}",
                 ((predict_point[0] + nearest_point[i][0])/2, (predict_point[1]+nearest_point[i][1])/2))


plt.show()
