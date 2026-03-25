import cv2
'''
外接轮廓
应用:1.图像分割 2.形状分析 3.物体检测识别s
主要函数 => 
    1.findContours() => 轮廓点
      drawContours() => 绘制轮廓
      外接矩形左上角坐标与矩形大小 = boundingRect(点集) => 找外接矩形
      rectangle() => 绘制矩形 
    
最小外接入矩形:旋转卡壳法,利用凸包
    1.随便找一条边作为基准,确定高度(与已知基准平行),然后确定宽度(投影最远处垂直线acosx)
    2.执行多次,使用最小的矩形
最小外接圆:welzl算法
    
'''
img = cv2.imread(r"D:\projects\Pylearning\image\contour.png")
img_draw = img.copy()
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

# 寻找轮廓
con, hie = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 绘制轮廓
cv2.drawContours(img_draw, con, -1, (0,0,255), 3)

# 绘制矩形(函数一次只能获取一个矩形)
for contour in con:
    x,y,w,h = cv2.boundingRect(contour)
    top_left = (x,y)
    bottom_right = (x+w,y+h)
    cv2.rectangle(img_draw, top_left, bottom_right, (255,0,0), 2)

# 显示结果
cv2.imshow("img",img)
cv2.imshow("img_draw",img_draw)
cv2.imshow("img_binary",img_binary)
cv2.waitKey(0)
cv2.destroyAllWindows()