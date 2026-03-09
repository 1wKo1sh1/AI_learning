import numpy as np
# 自定义协方差
x = np.array([[0, 2],
              [1, 1],
              [2, 0]])
# 平均值
x = x.T
x1=x[0,:]
x2=x[1,:]
mx1 = x1.mean()
mx2 = x2.mean()
n=x.shape[1]

x1x1 = np.sum((x1-mx1) ** 2) / (n-1)
x1x2 = np.sum((x1-mx1) * (x2-mx2)) / (n-1)
x2x1 = x1x2
x2x2 = np.sum((x2-mx2) ** 2) / (n-1)
cov = np.array([[x1x1, x1x2],
                [x2x1, x2x2]])
print("自定义:\n",cov)

# np库自带协方差函数
# 一个3*2的矩阵3样本2特征，但是计算需要2特征*3样本
x = np.array([[0, 2],
              [1, 1],
              [2, 0]])
'''X=
    x1, x2
    0, 2
    1, 1
    2, 0
'''
x = x.T
'''X=
    x1 0, 1, 2 
    x2 2, 1, 0
'''
# 输出为2*2的矩阵变成特征数量了即(x和y)
'''
    x1, x2
x1   1, -1
x2  -1, 1
'''
print("np自带:\n",np.cov(x))
