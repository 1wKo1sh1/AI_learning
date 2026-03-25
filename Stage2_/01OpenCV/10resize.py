import cv2

img = cv2.imread(r"D:\projects\Pylearning\image\_milkdragon.png")

#更换图像比例，填入dsize知道大小，填入fxfy知道比例且不同时为0,interpolation为插值方法
img_resized  = cv2.resize(img, dsize=None, fx=5, fy=0.8, interpolation=cv2.INTER_LINEAR)

# 显示结果
cv2.imshow("img",img)
cv2.imshow("img_resized",img_resized)
cv2.waitKey(0)
