import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
# datasets与torch.utils.data的dataset区别
from torch.utils.data import DataLoader, dataset
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from PIL import Image
import glob

# 设置随机种子以保证结果的可重复性
def setup_seed(seed):
    np.random.seed(seed)  # 设置 Numpy 随机种子
    random.seed(seed)  # 设置 Python 内置随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置 Python 哈希种子
    torch.manual_seed(seed)  # 设置 PyTorch 随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 设置 CUDA 随机种子
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False  # 关闭 cudnn 加速
        torch.backends.cudnn.deterministic = True  # 设置 cudnn 为确定性算法


# 设置随机种子
setup_seed(0)
# 检查是否有可用的 GPU，如果有则使用 GPU，否则使用 CPU
if torch.cuda.is_available():
    device = torch.device("cuda")  # 使用 GPU
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")  # 使用 CPU
    print("CUDA is not available. Using CPU.")

# =====================================代码本体====================================
# 1.数据处理

train_transform = transforms.Compose([
    # 保证最短边至少256后在进行随机裁剪
    transforms.Resize(256),
    # 先随机选择裁剪区域（面积比例在 [0.08, 1.0]，宽高比在 [3/4, 4/3]），然后缩放至 224×224。
    transforms.RandomResizedCrop(224),
    # 更强的随数据增强
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    # 以 50% 的概率水平翻转图像。
    transforms.RandomHorizontalFlip(0.5),
    # # 从图像中心裁剪一个 224×224 的正方形区域。
    # transforms.CenterCrop(224),
    # 将 PIL Image 或 NumPy 数组（H×W×C，值域 [0,255]）转换为 PyTorch 张量（C×H×W，值域 [0.0, 1.0]）。
    transforms.ToTensor(),
    # 对每个通道进行标准化：output = (input - mean) / std。
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    # 保证最短边至少256后在进行随机裁剪
    transforms.Resize(256),
    # 验证集不增加随机增强
    # transforms.RandomResizedCrop(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])
# 之前测试过fruits100 500mb的大小 训练速度很慢 ： https://www.modelscope.cn/datasets/tany0699/fruits100
# 训练集来源kaggle 15mb大小 ： https://www.kaggle.com/datasets/karimabdulnabi/fruit-classification10-class
train_dataset = datasets.ImageFolder('../dataset/train', transform=train_transform) # 230*10 占用70%
test_dataset = datasets.ImageFolder('../dataset/test', transform=val_transform) # 100*10 占用30%

class_names = test_dataset.classes # 获取类名

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# # 打印一下图片
# examples = enumerate(test_dataloader)
# batch_idx, (imgs, labels) = next(examples)
# fig = plt.figure()
# for i in range(4):
#     plt.subplot(2,2,i+1)
#     plt.imshow(imgs[i][0], cmap='gray')
#     plt.title(labels[i])
#     plt.xticks([])
#     plt.yticks([])
# plt.show()

# 2.加载ImageNet上的训练模型resnet50进行迁移训练
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# 替换最后一层为自己的10结果
num_features = model.fc.in_features # 获取ResNet50的原来的分类层
model.fc = nn.Linear(num_features, 10)  # 改为新的分类层10类
model = model.to(device)
model_max = model
# 冻结参数保护预训练模型已经学到的通用特征，只让新添加的分类层适应新任务
for param in model.parameters(): # 冻结所有卷积层
    param.requires_grad = False
for param in model.fc.parameters(): # 解冻分类层
    param.requires_grad = True

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

train_losses = []
val_losses = []
best_loss = float('inf')
patience_counter = 0
save_path = "../model/best_model.pth"
# 4.迭代封装到函数内
def train_model(model, criterion, optimizer, train_dataloader, test_dataloader, epochs=50, patience=2):
    global best_loss

    for epoch in range(epochs):
        # a.训练模式
        print(f":::Epoch [{epoch + 1}/{epochs}]:::")
        model.train()
        train_loss = 0
        for i, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)

            res = model(images)
            loss = criterion(res, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            print(f"\r<->Iter [{i}/{len(train_dataloader)}], Loss {loss:.4f}", end='',flush=True)

        train_loss = train_loss / len(train_dataloader)
        train_losses.append(train_loss)
        print(f"\nmodel.train : Loss {train_loss:.4f}")

        # b.验证模式
        model.eval()
        with torch.no_grad(): # 虽然不进行反向传播，但仍然会为中间变量构建计算图浪费计算，这样能加速并且降低显存占用
            eval_loss = 0
            for i, (images, labels) in enumerate(test_dataloader):
                images = images.to(device)
                labels = labels.to(device)

                res = model(images)
                loss = criterion(res, labels)
                # 该处为验证模式不进行反向传播优化
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()

                eval_loss += loss.item()
                # print(f"Epoch [{epoch + 1}/{epochs}], Iter [{i}/{len(test_dataloader)}], Loss {loss:.4f}")

            val_loss = eval_loss / len(test_dataloader)
            val_losses.append(val_loss)
            print(f"model.eval : Loss {val_loss:.4f}")

        # c.早停检测
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"   -> 保存最佳模型，验证损失: {best_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停触发，停止训练。最佳验证损失: {best_loss:.4f}")
                break
        print("===============================================")

    return model


# 最终验证模式
def evaluate_model(model, criterion, test_dataloader):
    model.load_state_dict(torch.load('../model/best_model.pth'))
    model = model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():  # 虽然不进行反向传播，但仍然会为中间变量构建计算图浪费计算，这样能加速并且降低显存占用
        eval_loss = 0
        for _, (images, labels) in enumerate(test_dataloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            # 该处为验证模式不进行反向传播优化
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            eval_loss += loss.item()

    """
            计算准确率、精确率、召回率、F1-score、混淆矩阵，并绘制曲线
    """
    avg_loss = eval_loss / len(test_dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    print("\n========== 模型评估结果 ==========")
    print(f"Loss             : {avg_loss:.4f}")
    print(f"Accuracy  (准确率): {accuracy:.4f}")
    print(f"Precision (精确率): {precision:.4f}")
    print(f"Recall    (召回率): {recall:.4f}")
    print(f"F1-score         : {f1:.4f}") #精确率和召回率的调和平均值，取值范围在 [0, 1] 之间越大表示模型性能越好
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.show()

def train(model, criterion, optimizer, train_dataloader, test_dataloader):
    # 开始训练
    model = train_model(model, criterion, optimizer, train_dataloader, test_dataloader, epochs=20, patience=2)

    # 解冻l4
    for name, param in model.named_parameters():
        if 'layer4' in name or 'fc' in name:
            param.requires_grad = True
            print(f"{name}: {param.requires_grad}")
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, weight_decay=1e-4)
    model = train_model(model, criterion, optimizer, train_dataloader, test_dataloader, epochs=20, patience=2)

    # 解冻l3 l4
    for name, param in model.named_parameters():
        if 'layer3' in name or 'layer4' in name or 'fc' in name:
            param.requires_grad = True
            print(f"{name}: {param.requires_grad}")
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-7, weight_decay=1e-4)
    model = train_model(model, criterion, optimizer, train_dataloader, test_dataloader, epochs=10, patience=2)
    return model

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
    model.load_state_dict(torch.load('../model/best_model.pth'))
    model = model.to(device)
    """交互菜单"""
    while True:
        print("\n" + "=" * 30)
        print("水果分类识别系统".center(20))
        print("=" * 30)
        print("1. "+"单张图片预测".center(20))
        print("2. "+"多张图片批量预测".center(20))
        print("3. "+"查看模型评估结果".center(20))
        print("4. "+"退出系统".center(20))
        print("5. "+"重新训练模型".center(20))
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
            evaluate_model(model, criterion, test_dataloader)

        elif choice == '4':
            print("现在退出系统")
            break

        elif choice == '5':
            print("现在开始训练模型")
            train(model, criterion, optimizer, train_dataloader, test_dataloader)

        else:
            print("无效输入，请重新选择。")

if __name__ == "__main__":

    # ui
    user_ui(model, class_names, device, val_transform)