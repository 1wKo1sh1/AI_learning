import cv2
import numpy as np

img = cv2.imread(r"D:\projects\Pylearning\image\fei.png")
#转换颜色空间
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#掩膜
#针对粉色区域 H S V 范围
# lowerb = np.array([0, 30, 150])
# upperb = np.array([25, 250, 255])
lowerb = np.array([0, 20, 120])       # H:0-10 (红色端)
upperb = np.array([10, 200, 255])
#使用inrange确定是否在区间内   rgb或者hsv      对应的数组范围
pink_mask = cv2.inRange(img_hsv, lowerb, upperb)
#位与操作  提取出粉色区域
#只有当掩膜对应位置为非零时，才会进行自己与自己的按位与运算。进行位与操作，提取出粉色区域
res = cv2.bitwise_and(img, img, mask=pink_mask)

cv2.imshow("img", img)
cv2.imshow("mask", pink_mask)
cv2.imshow("res", res)
cv2.waitKey(0)