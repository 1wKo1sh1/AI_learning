import cv2

img = cv2.imread("D:\projects\Pylearning\image\zhen.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#计算阈值的方法：平均值法      二值化方法：阈值法        blocksize:5              C:10
img_binary1 = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5,5)


cv2.imshow("img", img)
cv2.imshow("img_gray", img_gray)
cv2.imshow("img_binary1", img_binary1)
cv2.waitKey(0)

