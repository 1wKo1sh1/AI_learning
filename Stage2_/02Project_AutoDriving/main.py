
import hqyj_mqtt
import queue
import numpy as np
import base64
import cv2
import matplotlib.pyplot as plt


# 格式转化
def b642np(image):
    # image为字典，将字典的值变为字节bytes
    img_data = base64.b64decode(image['image'])
    # 字节转np
    img_np = np.frombuffer(img_data, dtype=np.uint8)
    # np转img
    img = cv2.imdecode(img_np,cv2.IMREAD_COLOR)
    return img

# 透视变换
def perspective_tf(img):

    img_size = (img.shape[1], img.shape[0])  # 高度*宽度
    dt_l = 43
    dt_r = 40
    dy = 20
    # cv2.line(img,(80, img_size[1]),(img_size[0] // 2 - dt_l, img_size[1] // 2 - dy),(0,0,255),1)
    # cv2.line(img,(450, img_size[1]),(img_size[0] // 2 + dt_r, img_size[1] // 2 - dy),(0,0,255),1)

    src = np.float32(
        [[80, img_size[1]],# 左下
         [450, img_size[1]],# 右下
         [img_size[0] / 2 + dt_r, img_size[1] / 2 - dy],# 右上
         [img_size[0] / 2 - dt_l, img_size[1] / 2 - dy]])# 左上
    dst = np.float32(
        [[img_size[0] / 4, img_size[1]],
         [img_size[0] * 3 / 4, img_size[1]],
         [img_size[0] *3 /4, 0],
         [img_size[0] / 4,0]]
    )

    # 透视变换矩阵
    tf = cv2.getPerspectiveTransform(src, dst)
    # 透视变换
    img_warp = cv2.warpPerspective(img,tf,img_size, flags=cv2.INTER_LINEAR)
    return img_warp

# 先膨胀 后腐蚀
def dilate_erode(img):
    kernel_size = 15
    kernel = np.ones((kernel_size,kernel_size),np.uint8)#膨胀核
    kernel_erode = np.ones((11,11),np.uint8)#腐蚀核5
    img_dilate = cv2.dilate(img,kernel,iterations = 1)
    # cv2.imshow('img_dilate',img_dilate)
    # cv2.waitKey(0)
    img_erode = cv2.erode(img_dilate,kernel,iterations = 1)
    return img_erode


# 梯度提取
def ex_line_g(img_warp):

    # 滤波消除马赛克
    img_Gaussian = cv2.GaussianBlur(img_warp, (5, 5), 1)
    # 灰度化
    img_gray = cv2.cvtColor(img_Gaussian, cv2.COLOR_BGR2GRAY)
    # sobel算子进行梯度计算
    res = cv2.Sobel(img_gray,-1,1,0)
    # 二值化
    ret,img_binary = cv2.threshold(res,127,255,cv2.THRESH_BINARY)
    # 因为还有毛边,进行形态学变换
    img_d_e = dilate_erode(img_binary)
    return img_d_e

def hlsSelect(img,thresh=(220,255)):

    # 转化为hls通道并且提取l（亮度）通道信息
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:,:,1] # 提取第一个通道，即l通道
    l_channel = l_channel / np.max(l_channel) * 255 # 映射到0，255范围内(/最大值变成[0,1]然后*255变成[0,255])

    # 大于低阈值，小于高阈值的像素设置为1
    binary_output = np.zeros_like(l_channel) # 全黑初始化
    binary_output[(l_channel>thresh[0]) & (l_channel<thresh[1])] = 1 # 满足条件的认为是白色
    return binary_output

def labSelect(img,thresh=(212,220)):

    # 右侧归零,转化格式
    img_copy = img.copy()
    img_copy[:,240:,:]=(0,0,0)
    lab = cv2.cvtColor(img_copy, cv2.COLOR_BGR2LAB)
    lab_b = lab[:,:,2]
    # 判断是否最大值大于100，超过才进行映射
    if np.max(lab_b)>100:
        lab_b = lab_b / np.max(lab_b) * 255

    # 根据阈值筛选
    binary_output = np.zeros_like(lab_b)
    binary_output[(lab_b>thresh[0]) & (lab_b<thresh[1])] = 1
    return binary_output


# 颜色提取
def ex_line_c(img_warp):

    # 提取白色车道线:HSL模型(对光照不敏感，例如光打到黄色上面会变得更白)，提高鲁棒性，但是不能分辨颜色
    hls_binary = hlsSelect(img_warp)
    # 提取黄色车道线:lab模型(b代表蓝黄分量，更好提取黄色信息，对光亮不敏感)，提高鲁棒性，但是不能分辨其他颜色
    lab_binary = labSelect(img_warp)

    # 合并, 其中一个为1则设置模板为1
    combined_binary = np.zeros_like(lab_binary)
    combined_binary[(hls_binary == 1) | (lab_binary ==1)] = 1
    # 膨胀腐蚀
    img_d_e = dilate_erode(combined_binary)
    return img_d_e

# 找目标车道线
def finding_line(img_d_e):
    # 关注下半部分减少计算量的同时防止检测到其他车道线
    img_size = (img_d_e.shape[1], img_d_e.shape[0]) #长img.shape[1]*高img.shape[0] => 行 * 列=img_size[0],imgsize[1]
    print(img_size)# 480,270
    bottom_part_sum = np.sum(img_d_e[img_size[1] // 2:,:],axis=0)
    plt.plot(bottom_part_sum)
    # 左右折半分别计算最大值所在索引
    midpoint = img_size[0] // 2 #480/2=240
    left_max_x = np.argmax(bottom_part_sum[:midpoint])
    right_max_x = np.argmax(bottom_part_sum[midpoint:]) + midpoint # 图像只剩一半，索引从0开始所以加上中点位置偏移
    print(f"左侧最大值索引:{left_max_x},右侧最大值索引:{right_max_x}")

    # 获取白色(非零)坐标，返回索引
    nonzero_index = np.array(img_d_e.nonzero())
    w_xs, w_ys = nonzero_index[1], nonzero_index[0]

    # 定义窗口用于辨别车道线和链接区域
    windows_num = 10
    window_height = img_size[1] //windows_num
    window_width = 50
    minpoint_num = 40

    # 初始化窗口位置
    left_x_current = left_max_x
    right_x_current = right_max_x

    # 创建空列表接受像素位置
    left_list = []
    right_list = []

    # 创建三通道图像用于展示小窗口寻路线过程
    img_wins = np.dstack((img_d_e,img_d_e,img_d_e))
    # 更新
    for window in range(windows_num):
        # 计算当前窗口上下边界y坐标
        win_y_high = img_size[1] - (window + 1) * window_height
        win_y_low = img_size[1] - window * window_height

        # 左路线的左右坐标,右路线的左右坐标
        leftwin_x_left = left_x_current - window_width
        leftwin_x_right = left_x_current + window_width
        rightwin_x_left = right_x_current - window_width
        rightwin_x_right = right_x_current + window_width

        # 显示滑动过程
        cv2.rectangle(img_wins,(leftwin_x_left,win_y_high),(leftwin_x_right,win_y_low),(0,255,0),2)
        cv2.rectangle(img_wins, (rightwin_x_left, win_y_high), (rightwin_x_right, win_y_low), (0, 255, 0),2)
        cv2.imshow('img_wins',img_wins)

        # 窗口内的白色元素(非零)，下文为对w_ys[i]和w_xs[i]检查找到非0元素所在的i，即当前点
        w_left_index = ((w_ys >= win_y_high) & (w_ys < win_y_low) &
                        (w_xs >= leftwin_x_left) & (w_xs < leftwin_x_right)).nonzero()[0]
        w_right_index = ((w_ys >= win_y_high) & (w_ys < win_y_low) &
                         (w_xs >= rightwin_x_left) & (w_xs < rightwin_x_right)).nonzero()[0]

        # 加入列表
        left_list.append(w_left_index)
        right_list.append(w_right_index)

        # 更新窗口
        if len(w_left_index) > minpoint_num:
            left_x_current = int(np.mean(w_xs[w_left_index]))
        else:# 如果左侧没有找到满足40白点的更新目标，则从右侧获得信息更新左侧
            if len(w_right_index) > minpoint_num:
                delta1 = int(np.mean(w_xs[w_right_index])) - right_x_current
                left_x_current += delta1

        if len(w_right_index) > minpoint_num:
            right_x_current = int(np.mean(w_xs[w_right_index]))
        else:
            if len(w_left_index) > minpoint_num:
                delta2 = int(np.mean(w_xs[w_left_index])) - left_x_current
                right_x_current += delta2



if __name__ == '__main__':

    # # 创建消息队列
    # q_mqtt_data = queue.Queue(5)
    # i = 1
    # # 1.mqtt客户端初始化，链接服务器，与场景建立通信
    # mqtt_client = hqyj_mqtt.MQTTClient('127.0.0.1',21883,'bb','aa',q_mqtt_data)
    #
    # while True:
    #     image = q_mqtt_data.get()
    #
    #     if 'image' in image:
    #         # 解析base64编码为opencv可以处理格式
    #         b642np(image)
    #         img = b642np(image)

    # 读取图像
    img = cv2.imread('./2.png')
    # 透视变换
    img_warp = perspective_tf(img)
    # 提取车道线 => 1.梯度提取 2.颜色提取
    img_line_g = ex_line_g(img_warp)
    # img_line_c = ex_line_c(img_warp)
    # 左右车道线分离与拟合
    finding_line(img_line_g)
    # 结果显示
    cv2.imshow('img',img)
    cv2.imshow('img_warp',img_warp)
    cv2.imshow('img_line_g',img_line_g)
    cv2.imshow('img_line_g/2',img_line_g[img.shape[0]//2:,:])
    plt.show()
    cv2.waitKey(0)