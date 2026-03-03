import numpy as np
import cv2
import matplotlib.pyplot as plt

#全0数组700*700*(三维颜色通道)，对应黑色图
#uint8:unsigned int无符号整数(0-255)
image = np.zeros((700,700,3),dtype=np.uint8)

blocksize = 100
for rows in range(0, 700, blocksize):
    for cols in range(0, 700, blocksize):
        image[rows, :, :] = (255,255,255)
        image[:, cols, :] = (255,255,255)
red_cube = ([0,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,0])
blue_cube = ([0,0,0,0,0,0,0],[0,0,0,0,0,1,0],[0,0,0,0,1,0,0],[0,0,0,0,0,0,0],[0,0,1,0,0,0,0],[0,1,0,0,0,0,0],[0,0,0,0,0,0,0])
for i in range(0,7):
    for j in range(0,7):
        if red_cube[i][j] == 1:
            image[i*blocksize+1:(i+1)*blocksize, j*blocksize+1:(j+1)*blocksize, :] = (0,0,255)
        if blue_cube[i][j] == 1:
            image[i*blocksize+1:(i+1)*blocksize, j*blocksize+1:(j+1)*blocksize, :] = (255,0,0)

#opencv中image是BGR，matplotlib中image是RGB,进行转换
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.subplot(224)
plt.imshow(image_rgb)
plt.title("Original Image")
# 关闭坐标轴
plt.axis('off')
#拆分彩色通道，获取原图三个像素值
#第一种办法，数组切片
r = image_rgb[:,:,0]  
g = image_rgb[:,:,1]
b = image_rgb[:,:,2]
# #第二种opencv函数split()
# b,g,r= cv2.split(image)

#新图像展示三通道
blue_channel = np.zeros_like(image_rgb)
green_channel = np.zeros_like(image_rgb)
red_channel = np.zeros_like(image_rgb)
#使用012=bgr的顺序填充通道，即opencv的默认通道顺序
blue_channel[:,:,0] = b
green_channel[:,:,1] = g
red_channel[:,:,2] = r
blue_channel_rgb = cv2.cvtColor(blue_channel, cv2.COLOR_BGR2RGB)
green_channel_rgb = cv2.cvtColor(green_channel, cv2.COLOR_BGR2RGB)
red_channel_rgb = cv2.cvtColor(red_channel, cv2.COLOR_BGR2RGB)

#131:一行三列布局，本图第一个位置
plt.subplot(221)
plt.imshow(blue_channel_rgb)
plt.title("Blue Channel")
plt.axis('off')

#132:一行三列布局，本图第二个位置
plt.subplot(222)
plt.imshow(green_channel_rgb)
plt.title("Green Channel")
plt.axis('off')

#133:一行三列布局，本图第三个位置
plt.subplot(223)
plt.imshow(red_channel_rgb)
plt.title("Red Channel")
plt.axis('off')

plt.show()
# #winname会覆盖，需要不同名字不同窗口
# cv2.imshow("image",image)
# #指定时间（毫秒级）获取按键，0表示无限等待，未输入则返回-1
# cv2.waitKey(0)
