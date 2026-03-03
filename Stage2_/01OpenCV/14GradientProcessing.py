import cv2
import numpy as np

img = cv2.imread(r"D:\projects\Pylearning\image\badapple.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


kernel_horizontal = np.array([[-1,0,1],
                             [-2,0,2],
                             [-1,0,1]])
kernel_vertical = kernel_horizontal.T
# 1.filter2D:图像, 深度-1与原图相同, 卷积核, 锚点, 类似偏置+B, 边界填充
img_filter2D = cv2.filter2D(img, -1, kernel_vertical)

# # 自动求导得到核值(返回计算空间图像导数的滤波器系数)
# kernelfilter = cv2.getDerivKernels()
# 2.Sobel:图像, 深度, dx, dy, ksize大小, 缩放比例, 增量(偏置), 边界填充
# 对于深度，如果输入为八位图像，即灰度图可能会导致截断产生，即左白右黑做差可能会导致结果小于0，此时该结果直接为0变成黑色
img_Sobel = cv2.Sobel(img, -1, 1, 0)

# 3.Laplacian:图像, 深度, ksize大小, 比例, 偏增量, 边界填充
img_lap = cv2.Laplacian(img, -1, ksize=3)

cv2.imshow("img",img)
cv2.imshow("img_filter2D",img_filter2D)
cv2.imshow("img_Sobel",img_Sobel)
cv2.imshow("img_lap",img_lap)
cv2.waitKey(0)