import os
import random
import numpy as np
import torch
import torch.nn as nn
from openpyxl.styles.builtins import total
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
from AlexNet import AlexNet

# 设置种子使结果可复现
def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)  # 固定哈希种子
    np.random.seed(seed)  # np种子
    random.seed(seed)  # py种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False  # 关闭cuddn加速
        torch.backends.cudnn.deterministic = True  # 设置cudnn为确定性算法


# 检查gpu，cuda是否可用
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available, Using GPU")
else:
    device = torch.device("cpu")
    print("CUDA is NOT available, Using CPU")

# 1数据处理
# 统一化为227像素图，然后进行归一化，最后做成零中心化和对称分布(映射到-1，1之中方便反向传播)
# 零中心化：将输入数据的均值变为 0。在神经网络反向传播中，零中心化的输入有助于避免梯度更新时的“之”字形震荡，让梯度下降更稳定、收敛更快。
# 对称分布：将数据范围控制在 [-1, 1] 之间。这与很多激活函数（如 Tanh）的输出范围匹配，可以防止因输入值过大导致梯度饱和（例如 Sigmoid 函数的梯度在两端趋近于 0）
transform = {
    "train" : transforms.Compose([transforms.RandomResizedCrop(227),transforms.ToTensor(),  #Resize(27,227)同样可以
                                  transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]),
    "test" : transforms.Compose([transforms.RandomResizedCrop(227),transforms.ToTensor(),
                                  transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
}
# transforms.totensor可以将原始pil数据转化为张量并且归一化
train_data = datasets.ImageFolder(root="./dataset/train", transform=transform["train"])
test_data = datasets.ImageFolder(root="./dataset/test", transform=transform["test"])
# 划分小段
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

# # 显示
# examples = enumerate(test_loader) # 获取迭代器减少内存消耗
# batch_idx, (imgs, labels) = next(examples)
# for i in range(4):
#     # 将处理的图像反向处理
#     mean = np.array([0.5,0.5,0.5])
#     std = np.array([0.5,0.5,0.5])
#     img_origin = imgs[i].numpy() * std[:,None,None] + mean[:,None,None]
#     # 图片转化为np数组，改变(宽 高 通道)的顺序，(3,247,247)=>(247,247,3)
#     img_origin = np.transpose(img_origin,(1,2,0))
#     plt.subplot(2,2,i+1)
#     plt.imshow(img_origin, interpolation="nearest")
#     plt.title(f"Truth:{labels[i]}")
# plt.show()

# 2初始化模型
model = AlexNet(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# 3开始训练
model.train()
epochs = 100
for epoch in range(epochs):
    total_loss = 0
    for batch_idx, (imgs, labels) in enumerate(train_loader):
        #  转移设备
        imgs, labels = imgs.to(device), labels.to(device)
        # 前向
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        total_loss += loss
        # 反向
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"------epoch : {epoch + 1}, Iter : {batch_idx}, loss : {loss:.4f}, epoch_progress : {batch_idx/len(train_loader)*100:.2f}%")
    avg_loss = total_loss / len(train_loader)
    print(f"epoch : {epoch + 1}, loss : {avg_loss:.4f}, progress : {epoch/epochs*100:.2f}%")

    if (epoch+1) % 10 == 0 and epoch != 0:
        torch.save(model.state_dict(), f'./model/model_{epoch+1}.pth')