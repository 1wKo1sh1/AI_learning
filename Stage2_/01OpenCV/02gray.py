import cv2
import numpy as np
import matplotlib.pyplot as plt

#读取图片，第一个图片地址，第二个读取方式，默认BGR彩色图
image_np = cv2.imread("D:\projects\Pylearning\image\image1.png")

#shape的三个参数分别是高度、宽度、通道数
image_shape = image_np.shape

#zero按照高度，宽度创建
image_gray1 = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)

#定义权重
weight_r = 0.299
weight_g = 0.587
weight_b = 0.114

#遍历彩色图，进行加权平均
for i in range(image_shape[0]):
    for j in range(image_shape[1]):
        #获取当前像素的BGR值
        b = image_np[i, j, 0]
        g = image_np[i, j, 1]
        r = image_np[i, j, 2]
        #进行加权平均
        gray = int(b * weight_b + g * weight_g + r * weight_r)
        #赋值给灰度图
        image_gray1[i,j] = gray

#直接平均
image_gray2 = np.zeros_like(image_gray1, dtype=np.uint8)
for i in range(image_shape[0]):
    for j in range(image_shape[1]):
        #获取当前像素的BGR值
        b = image_np[i, j, 0]
        g = image_np[i, j, 1]
        r = image_np[i, j, 2]
        #直接平均
        gray = int((b + g + r) / 3)
        #赋值给灰度图
        image_gray2[i,j] = gray
#最大值
image_gray3 = np.zeros_like(image_gray1, dtype=np.uint8)
for i in range(image_shape[0]):
    for j in range(image_shape[1]):
        #获取当前像素的BGR值
        b = image_np[i, j, 0]
        g = image_np[i, j, 1]
        r = image_np[i, j, 2]
        #最大值
        gray = int(max(b, g, r))
        #赋值给灰度图
        image_gray3[i,j] = gray



cv2.imshow("image_np", image_np)
cv2.imshow("gray_weight", image_gray1)
cv2.imshow("gray2_mean", image_gray2)
cv2.imshow("gray3_max", image_gray3)
cv2.waitKey(0)

