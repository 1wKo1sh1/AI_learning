import cv2
import numpy as np

img = cv2.imread(r"D:\projects\Pylearning\image\zhen.png")
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lowerb = np.array([0, 20, 120])      
upperb = np.array([10, 200, 255])

img_mask = cv2.inRange(img_hsv, lowerb, upperb)

#开运算(先腐蚀后膨胀) 闭运算(先膨胀后腐蚀)  形态学梯度(膨胀-腐蚀) 礼帽(原始-开运算) 黑帽(闭运算-原始)
#cv2.MORPH_OPEN:开运算;cv2.MORPH_CLOSE:闭运算;cv2.MORPH_GRADIENT:形态学梯度;cv2.MORPH_TOPHAT:礼帽;cv2.MORPH_BLACKHAT:黑帽
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
img_open = cv2.morphologyEx(img_mask, cv2.MORPH_OPEN, kernel)

#图片颜色替换（RGB空间下）
for i in range(img_mask.shape[0]):
    for j in range(img_mask.shape[1]):
        if img_mask[i, j] == 255:#如果此处掩膜为白色，代表原图为粉色
            img[i, j] = (0, 255, 0)#赋值为目标绿色

#显示结果
cv2.imshow("mask", img_mask)
cv2.imshow("open", img_open)
cv2.imshow("img", img)
cv2.waitKey(0)