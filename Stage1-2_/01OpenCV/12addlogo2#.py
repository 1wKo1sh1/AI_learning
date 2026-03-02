import cv2
# 读取两种图片
img = cv2.imread(r"D:\projects\Pylearning\image\_zimin1.png")
logo = cv2.imread(r"D:\projects\Pylearning\image\nanhun.png")

#logo大小
row,col = logo.shape[:2] #0，1维度为行和列
#截取原图ROI区域
start_row,start_col = 100,100 #logo插入位置
roi = img[start_row:start_row+row, start_col:start_col+col]

# 处理logo图像
logo_gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
ret,logo_binary = cv2.threshold(logo_gray, 127, 255, cv2.THRESH_BINARY_INV) #同理(原图为白色背景，转化为黑色背景)
# 与位运算（掩膜为白色区域直接显示
logo_and_logo = cv2.bitwise_and(logo, logo, mask=logo_binary)
# 融合图层 dst背景色为原始图，前景为logo，大小为roi区域
dst = cv2.add(logo_and_logo, roi)
# 替换原图ROI区域
img[start_row:start_row+row, start_col:start_col+col] = dst

# 显示结果
cv2.imshow("logo",logo)
cv2.imshow("roi",roi)
cv2.imshow("logo_binary",logo_binary)
cv2.imshow("logo_and_logo",logo_and_logo)
cv2.imshow("dst",dst)

cv2.imshow("img",img)

cv2.waitKey(0)