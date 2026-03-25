import json

import hqyj_mqtt
from pid import PID

import queue
import numpy as np
import base64
import cv2
import matplotlib.pyplot as plt
import time

class LaneCenterPlotter:
    def __init__(self, max_frames=200, img_height=480):
        # 设置plt为交互模式
        plt.ion()

        self.fig, self.ax = plt.subplots()
        # 元组解包法，获取元素，可以使用返回值加，解包或者发送的时候使用[0]
        self.line_lane_center, = self.ax.plot([], [], 'r-', label='lane_center')
        self.line_img_center, = self.ax.plot([], [], 'b-', label='img_center')

        # 设置图标
        self.ax.set_title('Lane and Img')
        self.ax.set_xlabel('Frame')
        self.ax.set_ylabel('Pixel Coordinate')
        self.ax.legend()

        self.x_data = []
        self.y_data_lane = []
        self.y_data_img = []

        self.max_frames = max_frames
        self.img_height = img_height

        # 初始化
        self.init_plot()

    # 初始化
    def init_plot(self):
        self.ax.set_xlim(0, self.max_frames)
        self.ax.set_ylim(0, self.img_height)
        self.line_lane_center.set_data([],[])
        self.line_img_center.set_data([],[])
        self.ax.grid()

    # 更新显示
    def update_plot(self, frame, lane_center, img_center):
        self.x_data.append(frame)
        self.y_data_lane.append(lane_center)
        self.y_data_img.append(img_center)

        # 更新
        self.line_lane_center.set_data(self.x_data, self.y_data_lane)
        self.line_img_center.set_data(self.x_data, self.y_data_img)

        # 保持x轴范围固定
        if len(self.x_data)>self.max_frames:
            self.ax.set_xlim(self.x_data[-self.max_frames], self.x_data[-1])
            self.ax.figure.canvas.draw()
        plt.pause(0.01)




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
    tf_inv = cv2.getPerspectiveTransform(dst, src)
    # 透视变换
    img_warp = cv2.warpPerspective(img,tf,img_size, flags=cv2.INTER_LINEAR)
    return img_warp,tf_inv

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
    # print(img_size)# 480,270
    bottom_part_sum = np.sum(img_d_e[img_size[1] // 2:,:],axis=0)
    # plt.plot(bottom_part_sum)
    # 左右折半分别计算最大值所在索引
    midpoint = img_size[0] // 2 #480/2=240
    left_max_x = np.argmax(bottom_part_sum[:midpoint])
    right_max_x = np.argmax(bottom_part_sum[midpoint:]) + midpoint # 图像只剩一半，索引从0开始所以加上中点位置偏移
    # print(f"左侧最大值索引:{left_max_x},右侧最大值索引:{right_max_x}")

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
        # cv2.rectangle(img_wins,(leftwin_x_left,win_y_high),(leftwin_x_right,win_y_low),(0,255,0),2)
        # cv2.rectangle(img_wins, (rightwin_x_left, win_y_high), (rightwin_x_right, win_y_low), (0, 255, 0),2)
        # cv2.imshow('img_wins',img_wins)

        # 窗口内的白色元素(非零)，下文为对w_ys[i]和w_xs[i]检查找到非0元素所在的i，即当前点
        w_left_index = ((w_ys >= win_y_high) & (w_ys < win_y_low) &
                        (w_xs >= leftwin_x_left) & (w_xs < leftwin_x_right)).nonzero()[0]
        w_right_index = ((w_ys >= win_y_high) & (w_ys < win_y_low) &
                         (w_xs >= rightwin_x_left) & (w_xs < rightwin_x_right)).nonzero()[0]

        # 加入列表（索引）
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

        # 记录上一次位置
        left_x_pre = left_x_current
        right_x_pre = right_x_current

    # print(type(left_list))
    # 连接索引列表，后续方便提取处像素点xy坐标以便拟合
    left_list = np.concatenate(left_list)
    right_list = np.concatenate(right_list)
    # print(type(left_list),right_list.shape)

    # 获得左侧右侧车道线像素位置,list为滑动窗口已经确认的白色点位置索引（上文计算的是索引），然后应用索引到白色像素数组获取点集
    leftx, lefty = w_xs[left_list], w_ys[left_list]
    rightx, righty = w_xs[right_list], w_ys[right_list]

    # 使用np.ployfit（因变量，自变量，多项式次数）拟合
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # 绘制图像的横坐标取值空间
    ploty = np.linspace(0, img_size[1] - 1, img_size[1])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # 中间车道
    middle_fitx = (left_fitx + right_fitx) // 2

    # 不同颜色绘制
    img_wins[lefty,leftx] = [255,0,0]
    img_wins[righty,rightx] = [0,0,255]
    # cv2.imshow("img_out",img_wins)

    return left_fitx, right_fitx, middle_fitx, ploty

def show_line(img, img_warp, img_line_g, tf_inv, left_fitx, right_fitx, middle_fitx, ploty):
    # 原图像，变换结果，车道线在原图的结果，单纯车道线
    img0 = np.zeros_like(img_line_g).astype(np.uint8)
    img_3c = np.dstack((img0,img0,img0))

    # 组合x和y坐标left_fitx, right_fitx, ploty长度都是270
    pts_left = np.transpose(np.vstack([left_fitx, ploty])) # 原形状为2*270应该为270*2即有270个点，分别使用xy坐标表示
    pts_right = np.transpose(np.vstack([right_fitx, ploty]))
    pts_middle = np.transpose(np.vstack([middle_fitx, ploty]))

    # 绘制
    cv2.polylines(img_3c, np.int32([pts_left]), isClosed = False, color=(202,124,0), thickness=15)
    cv2.polylines(img_3c, np.int32([pts_right]), isClosed=False, color=(202, 124, 0), thickness=15)
    cv2.polylines(img_3c, np.int32([pts_middle]), isClosed=False, color=(202, 124, 0), thickness=15)
    # cv2.imshow('img_3c',img_3c)

    # 逆变换映射回去
    img_inv = cv2.warpPerspective(img_3c, tf_inv, (img.shape[1],img.shape[0]))
    # 和原图融合的原图结果车道线
    img_imginv = cv2.addWeighted(img,1,img_inv,1,0,)

    # 单纯车道线
    bg = np.zeros_like(img).astype(np.uint8) + 127
    img_line = cv2.addWeighted(bg,1,img_inv,1,0,)

    # 结果显示
    # cv2.imshow('img_imginv', img_imginv)
    # cv2.imshow('img', img)
    # cv2.imshow('img_line', img_line)
    # cv2.imshow('img_warp', img_warp)
    result1 = np.concatenate((img,img_warp),axis=1)
    result2 = np.concatenate((img_line,img_imginv),axis=1)
    result = np.concatenate((result1,result2),axis=0)
    cv2.imshow("result",result)
    cv2.waitKey(1)
    return pts_middle

# 自动驾驶(图片，mqtt客户端交互，中间道路，pid，)
def auto_run(img, mqtt_client,  pts_middle, pid, carspeed=20):
    # 中心车道线中心坐标，目标位置(其实为当前位置，模型中目标与当前坐标互换)
    lane_center = pts_middle[240:,:].mean()
    # 当前位置（其实为目标位置）
    img_center = img.shape[1] // 2

    steering_angle = -pid(lane_center)

    # 输出控制指令
    mqtt_client.send_mqtt(json.dumps({"carSpeed":carspeed}))
    mqtt_client.send_mqtt(json.dumps({"carDirection": steering_angle}))
    print(f"输出角度{steering_angle}")
    return lane_center, img_center


if __name__ == '__main__':
    # 创建消息队列
    q_mqtt_data = queue.Queue(5)

    # 实例化plt辅助工具
    frame = 0
    plotter = LaneCenterPlotter()

    # 1.mqtt客户端初始化，链接服务器，与场景建立通信
    mqtt_client = hqyj_mqtt.MQTTClient('127.0.0.1',21883,'bb','aa',q_mqtt_data)

    # # 计算帧率
    # frame_count = 0
    # start_time = time.time()

    # 初始化pid
    pid = PID(Kp=0.3, Ki=0.01, Kd=0.1, setpoint=240)
    pid.sample_time = 0.1
    pid.output_limits = (-13,13)


    while True:
        try:
            image = q_mqtt_data.get()

            if 'image' in image:

                # # 计算帧率
                # frame_count += 1
                # current_time = time.time()
                # fps = frame_count / (current_time - start_time) if (current_time - start_time) else 0
                # print(f"FPS: {fps}")

                # 解析base64编码为opencv可以处理格式
                b642np(image)
                img = b642np(image)

                # 读取图像
                # img = cv2.imread('./7.png')
                # 透视变换
                img_warp,tf_inv = perspective_tf(img)
                # 提取车道线 => 1.梯度提取 2.颜色提取
                img_line_g = ex_line_g(img_warp)
                # img_line_c = ex_line_c(img_warp)
                # 左右车道线分离与拟合
                left_fitx, right_fitx, middle_fitx, ploty = finding_line(img_line_g)
                # 绘制车道线
                pts_middle = show_line(img, img_warp, img_line_g, tf_inv, left_fitx, right_fitx, middle_fitx, ploty)

                # 自动驾驶
                lane_center, img_center = auto_run(img, mqtt_client, pts_middle, pid)
                # 实时显示误差
                plotter.update_plot(frame, lane_center, img_center)
                frame += 1


        except Exception as e:
            print(e)