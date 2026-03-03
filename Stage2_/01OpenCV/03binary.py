import cv2
import numpy as np

img = cv2.imread("D:\projects\Pylearning\image\image1.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

thresh = 127 #阈值
maxval = 255 #最大值
#使用cv2函数二值化
#ret是阈值，img_binary是二值化后的图像,使用阈值法等方法不会返回阈值没有作用
#ret在使用OTSU方法计算阈值才会有用
ret1, img_binary1 = cv2.threshold(img_gray, thresh, maxval, cv2.THRESH_BINARY)    #二值化 阈值以下为0，以上为255
ret2, img_binary2 = cv2.threshold(img_gray, thresh, maxval, cv2.THRESH_BINARY_INV) #反二值化 阈值以下为255，以上为0
ret3, img_binary3 = cv2.threshold(img_gray, thresh, maxval, cv2.THRESH_TRUNC)      #截断 阈值以下为原数值，以上为阈值
#OTSU
ret4, img_binary4 = cv2.threshold(img_gray, thresh, maxval, cv2.THRESH_BINARY + cv2.THRESH_OTSU)        #OTSU 自动计算阈值
print(ret4)
'''
#一般计算原理
img_shape = img_gray.shape #高度x宽度x通道数
#创建单通道图像存储比较结果
img_binary_a = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)

#遍历灰度图里面的像素值比较
for i in range(img_shape[0]):
    for j in range(img_shape[1]):
        if img_gray[i,j] > thresh:
            img_binary_a[i,j] = maxval
        else:
            img_binary_a[i,j] = 0 
'''

'''
#OTSU原理
#获取数组最小值，最大值
start_thresh = img_gray.min()
end_thresh = img_gray.max()
#获取灰度图高度宽度
img_shape= img_gray.shape

#定义最大类间方差公式所需变量
rows = img_shape[0]
cols = img_shape[1]

#定义字典存储每一个阈值的最大类间方差
var= {}

#开始遍历阈值
for T in range(start_thresh+1, end_thresh, 1):
    #使用NumPy的布尔索引直接计算前景和背景像素,避免溢出
    foreground = img_gray > T  # 前景像素
    background = img_gray <= T  # 背景像素
    
    #计算区域内像素值的数量
    n_0 = np.sum(foreground)
    n_1 = np.sum(background)
    
    #避免除以零
    if n_0 == 0 or n_1 == 0:
        continue
    
    #计算区域内像素值的权重，即前景占全局像素比例
    w_0 = n_0 / (rows * cols)
    w_1 = n_1 / (rows * cols)
    
    #计算区域内像素值的平均值,使用NumPy的sum避免溢出
    u_0 = np.sum(img_gray[foreground]) / n_0  # 前景平均值
    u_1 = np.sum(img_gray[background]) / n_1  # 背景平均值
    
    #整张图平均像素值
    u = np.mean(img_gray)
    
    #计算最大类间方差
    g = w_0 * (u_0 - u) ** 2 + w_1 * (u_1 - u) ** 2
    
    #存入字典方便后续选出最大值
    var[T] = g

#选出字典中最大值中的键
best_thresh = max(var, key=var.get)

img_binary_b = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)
#遍历灰度图里面的像素值比较
for i in range(img_shape[0]):
    for j in range(img_shape[1]):
        if img_gray[i,j] > best_thresh:
            img_binary_b[i,j] = maxval
        else:
            img_binary_b[i,j] = 0 
'''


cv2.imshow("img", img)
cv2.imshow("img_gray", img_gray)
cv2.imshow("img_binary1", img_binary1)
cv2.imshow("img_binary2", img_binary2)
cv2.imshow("img_binary3", img_binary3)
cv2.imshow("img_binary4", img_binary4)
# cv2.imshow("img_binary_b", img_binary_b)

cv2.waitKey(0)