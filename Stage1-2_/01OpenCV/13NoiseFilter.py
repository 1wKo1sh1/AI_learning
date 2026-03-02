import cv2
'''
卷积核(滤波器)在原图上滑动计算 => 结果为滤波结果;核值不同结果不同
均值滤波:1/格数,卷积核每格均分
方框滤波:比均值多了normalize选项,true则与均值相同,false则单位化(变亮)
中值滤波:无核值,对覆盖范围内像素值排序,找到中位数作为滤波结果;很适合椒盐噪声(0和255点很多,点周围全是黑或者白才能产生黑白结果，如果一半没有黑和白，很大概率为正常像素值)
高斯滤波:类似自适应二值化,使用高斯函数生成高斯核更平滑处理,多了sigmaX参数越大模糊越明显,噪点消失效果更明显,更慢但是效果好
双边滤波:最复杂但是最好,一般滤波导致边缘信息消失,高斯为根据位置得到权重(空域)，而双边加了对像素值权重(值域),使用两个高斯公式
        两个sigma参数:不想考虑令他们相等,小于10几乎无影响,大于150卡通化
使用:不知道用什么 => 高斯 ; 斑点和椒盐 => 中值 ; 尽可能保留边缘信息 => 双边 ;
    线性(快):均值,方框,高斯 ; 非线性(慢):中值,双边
'''

# 展示高斯滤波(模糊)
img = cv2.imread(r"D:\projects\Pylearning\image\zhen.png")
# # 自动求导得到核值(返回计算空间图像导数的滤波器系数):核大小,sigma计算方式,滤波系数类型
# kernelfilter = cv2.getDerivKernels()
# GaussianBlur:图像, 核, sigmax, sigmay默认等于sigmax, 边界填充默认边界反射101
img_gauss = cv2.GaussianBlur(img,(3,3),1)

# 展示双边滤波
# bilateralFilter:图像, 滤波器大小d, sigmacolor(越大考虑范围大), sigmaspace(越大越远的像素影响越大 当d>0时不管space多少都确定领域的大小), 边界填充(无限制可以使用border_wrap)
img_bilateral = cv2.bilateralFilter(img,5,170,170)

cv2.imshow("img",img)
cv2.imshow("img_gauss",img_gauss)
cv2.imshow("img_bilateral",img_bilateral)
cv2.waitKey(0)
