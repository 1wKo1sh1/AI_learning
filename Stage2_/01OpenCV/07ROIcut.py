import cv2
import numpy as np
#np中为先高后宽(即行列)，opencv中为先宽后高(即列行)

img = cv2.imread(r"D:\projects\Pylearning\image\_milkdragon.png")

#获取图片高宽
height,width = img.shape[0], img.shape[1]

try:
    #切割区域
    xm, ym = 153, 141
    xM, yM = 280, 235

    #判断区域是否合理
    if (xm < 0 or xM > width or ym < 0 or yM > height):
        raise OverflowError("切割区域超出图片范围")

    #用np操作切割区域
    img_roi = img[ym:yM, xm:xM] 

    #绘制矩形框观察区域
    cv2.rectangle(img, (xm-2,ym-2), (xM+2,yM+2), (0,0,255),2)

    #显示结果
    cv2.imshow("img", img)
    cv2.imshow("roi", img_roi)
    cv2.waitKey(0)

except OverflowError as e:
    print(e)

        


