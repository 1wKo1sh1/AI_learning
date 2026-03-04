import hqyj_mqtt
import queue
import numpy as np
import base64
import cv2

def b642np(image):
    # image为字典，将字典的值变为字节bytes
    img_data = base64.b64decode(image['image'])
    # 字节转np
    img_np = np.frombuffer(img_data, dtype=np.uint8)
    # np转img
    img = cv2.imdecode(img_np,cv2.IMREAD_COLOR)
    return img

if __name__ == '__main__':

    # 创建消息队列
    q_mqtt_data = queue.Queue(5)
    i = 1
    # 1.mqtt客户端初始化，链接服务器，与场景建立通信
    mqtt_client = hqyj_mqtt.MQTTClient('127.0.0.1',21883,'bb','aa',q_mqtt_data)

    while True:
        image = q_mqtt_data.get()

        if 'image' in image:
            # 解析base64编码为opencv可以处理格式
            b642np(image)
            img = b642np(image)

            cv2.imshow('img',img)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f'{i}.png', img)
                i += 1
                print('save successful')
            else:
                continue