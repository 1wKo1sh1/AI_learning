import cv2 
import numpy as np

img = cv2.imread(r"D:\projects\Pylearning\image\cube.png")

# 原图像大小(np中为先高后宽(即行列)
height,width = img.shape[0],img.shape[1]

# 原图像四个顶点(左上、右上、左下、右下)
pts1 = np.float32([[269,82], [698,88], [25,522], [917,536]])
# 目标图像四个顶点
pts2 = np.float32([[0,0], [width,0], [0,height], [width,height]])

#原图像四个点
p0 = pts1[0].astype(np.int64).tolist()
p1 = pts1[1].astype(np.int64).tolist()
p2 = pts1[2].astype(np.int64).tolist()
p3 = pts1[3].astype(np.int64).tolist()
#绘制线段方便观察
cv2.line(img, p0, p1, (0,0,255), 1, lineType=cv2.LINE_AA) #绘制方法1:直接填写坐标
cv2.line(img, p2, p3, (0,0,255), 1, lineType=cv2.LINE_AA)  #绘制方法2:pts1[0].astype(np.int64).tolist()
cv2.line(img, p0, p2, (0,0,255), 1, lineType=cv2.LINE_AA)
cv2.line(img, p1, p3, (0,0,255), 1, lineType=cv2.LINE_AA)

# 求解从pts1到2的透视变换矩阵(求解系统的最小二乘问题)
# solveMethod默认cv2.DECOMP_LU即高斯消元
M = cv2.getPerspectiveTransform(pts1, pts2, solveMethod=cv2.DECOMP_LU)

# 应用透视变换
# 插值方法flags:默认cv2.INTER_LINEAR(和WARP_INVERSE_MAP一起使用则使用逆矩阵) ; 边界填充方法borderMode:默认cv2.BORDER_CONSTANT ;填充值borderValue默认0
img_warp = cv2.warpPerspective(img, M, (width,height))


cv2.imshow("img",img)
cv2.imshow("img_warp",img_warp)
cv2.waitKey(0)



