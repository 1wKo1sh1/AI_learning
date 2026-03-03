import cv2
'''
边缘检测  Canny算法
1.高斯滤波(防止噪点干扰边缘)(但是加粗不明显的边界)
2.获取梯度大小与方向
3.非极大值抑制(验证点的梯度值是周围梯度的最大值)
4.双阈值筛选(对结果进一步筛选)(设置两个阈值一个小一个大，小的舍弃，大的设为强边缘，根据强边缘链接找弱1边缘)

用高阈值确定一批最可靠的“种子点”或“核心事件”。
用低阈值划定一个更大的“候选区”。
通过特定的连接规则（如空间邻接、逻辑关联、时序连续性），将候选区中与核心点相关联的点吸纳进来，形成最终结果。
这种方法巧妙地解决了单一阈值“非黑即白”判断的局限性，是许多高级算法中用于提升结果质量的关键步骤。
'''
img = cv2.imread(r"D:\projects\Pylearning\image\zhen.png")
# 灰度化
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 二值化测试(可关闭)
ret, img_binary = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)
# 高斯滤波
img_noise = cv2.GaussianBlur(img_binary,(5,5),1.5)
# 计算梯度方向(曼哈顿距离:L1,欧氏距离:L2)
# img, 阈值1, 阈值2, sobel算子大小, 默认使用l1(false)
edges = cv2.Canny(img_noise, 30, 70,)


# 结果
cv2.imshow("img",img)
cv2.imshow("edges",edges)
cv2.waitKey(0)