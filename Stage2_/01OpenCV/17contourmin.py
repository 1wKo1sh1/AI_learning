import cv2
import numpy as np
'''
外接轮廓
应用:1.图像分割 2.形状分析 3.物体检测识别s
主要函数 => 
     2.[角度,中心,宽度高度] = minAreaRect() => 找最小矩形
       coxPoints() => 查找旋转矩形四个顶点
       np.int64() => 对矩阵每个元素取整(64位系统)
       drawContours() => 绘制(矩形)轮廓
     3.圆心,半径 = minEnclosingCircle() => 找最小圆
       circle(画位置,圆心,半径,颜色,粗细,线类型) => 绘制圆   
     
最小外接入矩形:旋转卡壳法,利用凸包
    1.随便找一条边作为基准,确定高度(与已知基准平行),然后确定宽度(投影最远处垂直线acosx)
    2.执行多次,使用最小的矩形
最小外接圆:welzl算法

'''
img = cv2.imread(r"D:\projects\Pylearning\image\contour.png")
img_draw = img.copy()
img_draw2 = img.copy()
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

# 寻找轮廓
cnts, hie = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 绘制轮廓
cv2.drawContours(img_draw, cnts, -1, (0,0,255), 3)

# 最小外接矩形(依旧一个一个处理)
for cnt in cnts:
    #  使用一个变量接受方便使用函数
    rect = cv2.minAreaRect(cnt)
    # 结果为浮点数,取整
    box = np.int64(cv2.boxPoints(rect))
    # contours必须是cnts级别的,而box为cnt级别，所以需要外加括号提高级别
    cv2.drawContours(img_draw2, [box], -1, (255,0,0))


#  最小圆(依旧一个一个处理)
for cnt in cnts:
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    (x,y,radius) = np.int64((x,y,radius))
    cv2.circle(img_draw,(x,y),radius,(255,0,0))


cv2.imshow("img", img)
cv2.imshow("img_draw", img_draw)
cv2.imshow("img-draw2", img_draw2)
cv2.waitKey(0)