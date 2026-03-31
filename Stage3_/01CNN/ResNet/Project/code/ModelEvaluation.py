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
    transforms.RandomResizedCrop(224),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder('../dataset/train', transform=train_transform)
val_dataset = datasets.ImageFolder('../dataset/val', transform=val_transform)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

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
# 使用ResNet18模型
model = ResNet(BasicBlock, [2,2,2,2], num_classes=10).to(device)
cri = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

save_path = '../model/last.pth'
model.load_state_dict(torch.load(save_path))
print("评估模型")
model.eval()

total = 0
correct = 0
predicted_labels = []
true_labels = []
with torch.no_grad():
    for imgs, labels in test_dataloader:
        imgs,labels = imgs.to(device),labels.to(device)

        # 得到一个概率序列
        outputs = model(imgs)
        # 获取序列最大值，即目前概率最大的可能所在的索引
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().cpu().numpy()    # correct += (predicted == labels).sum().item()
        predicted_labels.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
# 正确率
print(f"ACC {correct / total * 100 : .2f}%")
# 混淆矩阵
conf = confusion_matrix(true_labels, predicted_labels)
# 热力图
sns.heatmap(conf, annot=True, fmt="d",cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()