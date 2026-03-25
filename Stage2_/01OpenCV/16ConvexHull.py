import cv2
"""
凸包:完全凸出没有凹处的多边形,即凸多边形。一个凸包伴随着一个点集存在
凸包主要用于物体识别,手势识别,边界检测
对于一个点集,如果该点存在凸包,那么这个点要么在凸包上要么在凸包内

算法：
    穷举       => 两两配对,计算其他点是否都在同侧,一边有点代表这俩点是凸包点;向量计算然后x乘(|a||b|sinx通过夹角判断)
    
    Graham扫描 => 1.纵坐标最小的点为p0,作为原点构建坐标系,p0一定为凸包点
                 2.计算各个点关于p0的角度从小到大排序(逆时针),角度同近点放前面,此时角度最大和最小的一定为凸包点(放射形状包括上面点)
                 3.栈记录p0p1入栈,继续找下一个点
                 4.入栈下一个点,栈顶元素相连,下一个点如果在右侧[步骤5],在左侧或者直线上[步骤6]
                 5.右侧代表不是凸包点,栈顶元素出栈,继续[步骤4]
                 6.左侧或者直线上代表是凸包点,保存该点
                 7.[步骤4]一直执行到栈顶元素为最大点
                 
    Andrew扫描链 => 1.x为第一关键字,y为第二关键字排序(从左到右顺序,同x这按照y从下到上),第一个最后一个肯定在凸包上
                   2.顺序 遍历所有点,三个一组进行x乘,判断是不是凸包点(下凸包)
                   3.逆序 如上,判断十倍速凸包点(上凸包)
                   
    QuikHull法  => 1.所有点放入坐标系,链接横坐标最小最大两点p1p2,分为两部分上包和下包
                   2.一边为例,找到距离分割线最远的,与断点连线,对于两条新产生的线,如果外面有点则继续执行,没点了就是凸包点,一直执行到全部包括在内 
"""

img = cv2.imread(r"D:\projects\Pylearning\image\ConvexHull.png")
img_copy = img.copy()
img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
ret, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

# 获取轮廓点坐标
# (点坐标,点关系) = findContours:图像(灰度二值化), 轮廓检索模式=最外层轮廓, 轮廓逼近方法=压缩水平垂直对角的冗余点, 只保留端点, 轮廓点的偏移量
contours,hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 查找凸包(根据轮廓点)
cnt = contours[0] #代表第一层轮廓的点坐标
# 凸包 = convexHull(基于sklansky算法) : 点集, 方向信号true为顺, 操作信号true返回凸包点false返回索引
img_hull = cv2.convexHull(cnt)

# 结果(使用函数绘制多边形曲线)
# 图像 = polylines:目标图像, 顶点集合, 是否闭合, 颜色, 粗细-1填充, 线条类型默认LINE_8(LINE_AA抗锯齿), 小数点位数
img_poly = cv2.polylines(img_copy, [img_hull], True, (0, 0, 255),2)

cv2.imshow("img",img)
cv2.imshow("img_binary",img_binary)
cv2.imshow("img_poly",img_poly)
cv2.waitKey(0)