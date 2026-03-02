import cv2 

img = cv2.imread(r"D:\projects\Pylearning\image\_zimin1.png")

#镜像翻转(0为水平翻转，+大于0为垂直翻转，-小于0为水平垂直翻转)
img_mirror = cv2.flip(img,1)

# 显示结果
cv2.imshow("img",img)
cv2.imshow("img_mirror",img_mirror)
cv2.waitKey(0)
