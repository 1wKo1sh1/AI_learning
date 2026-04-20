import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from PIL import Image
import os
import glob

def load_image(image_path, transform):
    """加载并预处理单张图片（RGB）"""
    # 使用PIL库进行快速转化格式
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # 添加 batch 维度

def predict_single(model, image_tensor, class_names, device):
    """单张图片预测，返回 (类别, 置信度)"""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probabilities, 1)
        return class_names[pred.item()], conf.item()

def predict_batch(model, folder_path, transform, class_names, device, extensions=('*.jpg', '*.jpeg', '*.png')):
    """批量预测文件夹内所有图片，打印每张图片的预测结果"""
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
        image_paths.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    if not image_paths:
        print(f"在 {folder_path} 中未找到图片文件")
        return
    print(f"\n找到 {len(image_paths)} 张图片，开始预测...")
    for img_path in image_paths:
        img_tensor = load_image(img_path, transform)
        pred_class, conf = predict_single(model, img_tensor, class_names, device)
        print(f"{os.path.basename(img_path)} -> {pred_class} ({conf:.4f})")
    print("批量预测完成。")


def user_ui(model, class_names, device, transform):
    """交互菜单"""
    while True:
        print("\n" + "=" * 30)
        print("  水果分类识别系统")
        print("=" * 30)
        print("1. 单张图片预测")
        print("2. 多张图片批量预测")
        print("3. 查看模型评估结果（测试集）")
        print("4. 退出系统")
        choice = input("请输入功能编号：").strip()
        if choice == '1':
            img_path = input("输入待预测图片路径：").strip()
            if not os.path.exists(img_path):
                print("不存在，检查路径")
                continue
            img_tensor = load_image(img_path, transform)
            pred_class, conf = predict_single(model, img_tensor, class_names, device)
            print(f"预测结果：{pred_class}")
            print(f"置信度：{conf:.4f}")
        elif choice == '2':
            folder = input("输入批量预测文件夹路径：").strip()
            if not os.path.isdir(folder):
                print("不存在，检查路径")
                continue
            predict_batch(model, folder, transform, class_names, device)
        elif choice == '3':

            pass
        elif choice == '4':
            print("现在退出系统")
            break
        else:
            print("无效输入，请重新选择。")