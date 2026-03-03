import cv2
# opencv中逆时针为正角度，顺时针为负角度
img = cv2.imread(r"D:\projects\Pylearning\image\_zimin1.png")

# 原图像大小(np中为先高后宽(即行列)，opencv中为先宽后高(即列行)
height, width = img.shape[0], img.shape[1] #np
# 定义角度和缩放比例
center = (width//2, height//2) #opencv中中心坐标为(width//2, height//2)
angle = 45
scale = 0.3

# 构建旋转,缩放,平移矩阵
M = cv2.getRotationMatrix2D(center, angle, scale)
# 逆变换
M_inv = cv2.invertAffineTransform(M)
# 应用矩阵( 原图像, 变换矩阵, 输出图像大小, 插值方法(双线性插值), 边界填充方法(常量填充), 填充值)
img_rotated = cv2.warpAffine(img, M, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0)) #cv
img_rotated2 = cv2.warpAffine(img, M, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) #cv
img_rotated_2 = cv2.warpAffine(img_rotated2, M, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) #cv
# 显示结果
cv2.imshow("img",img)
cv2.imshow("img_rotated",img_rotated)
cv2.imshow("img_rotated2",img_rotated2)
cv2.imshow("img_rotated_2",img_rotated_2)
cv2.waitKey(0)

