# 导入所需的库
import os
import random

# 导入数据处理和可视化库
import matplotlib.pyplot as plt
import numpy as np

# 导入深度学习框架 PyTorch 相关库
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ResNet import ResNet,BasicBlock

from sklearn.metrics import confusion_matrix
import seaborn as sns


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


# 数据处理

train_transform = transforms.Compose([
    # 保证最短边至少256后在进行随机裁剪
    transforms.Resize(256),
    # 先随机选择裁剪区域（面积比例在 [0.08, 1.0]，宽高比在 [3/4, 4/3]），然后缩放至 224×224。
    transforms.RandomResizedCrop(224),
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
    transforms.RandomResizedCrop(224),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder('../dataset/train', transform=train_transform)
test_dataset = datasets.ImageFolder('../dataset/test', transform=val_transform) # 用于测试输出指标

train_dataloader = DataLoader(train_dataset, batch_size=50, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=50, shuffle=False)

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

# 定义模型
model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=100).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

save_path = '../model/last.pth'

# 4.迭代
def train_model(model, criterion, optimizer, train_dataloader, test_dataloader, epochs=50):
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

        avg_loss = train_loss / len(train_dataloader)
        print(f"\nmodel.train : Loss {avg_loss:.4f}")

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

            avg_loss = eval_loss / len(test_dataloader)
            print(f"model.eval : Loss {avg_loss:.4f}")

        print("===============================================")

    return model

# 开始训练
model = train_model(model, criterion, optimizer, train_dataloader, test_dataloader, epochs=20)
