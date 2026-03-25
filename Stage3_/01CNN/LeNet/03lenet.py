import os
import random
import numpy as np
import torch
import torch.nn as nn
from openpyxl.styles.builtins import total
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
from lenet5 import LeNet5

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

# transforms.totensor可以将原始pil数据转化为张量并且归一化
train_data = datasets.MNIST(root="./dataset", train=True, transform=transforms.ToTensor(), download=True)
test_data = datasets.MNIST(root="./dataset", train=False, transform=transforms.ToTensor(), download=True)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

# # 显示
# examples = enumerate(test_loader)
# batch_idx, (imgs, labels) = next(examples)
# for i in range(6):
#     plt.subplot(2,3,i+1)
#     plt.imshow(imgs[i][0], cmap="gray", interpolation="nearest")
#     plt.title(f"Truth:{labels[i]}")
# plt.show()

model = LeNet5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

epoches = 10
model.train()
for epoch in range(epoches):
    total_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print(f"epoch : {epoch+1}, loss : {avg_loss:.4f}")

torch.save(model.state_dict(), 'model_lenet5.pth')