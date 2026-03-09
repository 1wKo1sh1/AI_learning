import numpy as np
import matplotlib.pyplot as plt
from sipbuild.generator.parser.rules import p_include

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
Y = np.concatenate((np.zeros_like(len(class1_points), np.ones_like(len(class1_points))),))

# 2.计算先验
n = class1_points.shape[0]+class2_points.shape[0]
p_per1 = class1_points/n
p_per2 = class2_points/n
