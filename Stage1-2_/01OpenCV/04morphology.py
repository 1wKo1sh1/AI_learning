import cv2

img = cv2.imread("D:\projects\Pylearning\image\zhen.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret1, img_binary1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

#腐蚀(腐蚀白色)
#MORPH_RECT:矩形结构元素;MORPH_CROSS:十字结构元素;MORPH_ELLIPSE:椭圆结构元素
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
img_erode = cv2.erode( img_binary1, kernel)

#膨胀(膨胀白色)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
img_dilate = cv2.dilate( img_binary1, kernel)

cv2.imshow("img_binary1", img_binary1)
cv2.imshow("img_erode", img_erode)
cv2.imshow("img_dilate", img_dilate)
cv2.waitKey(0)