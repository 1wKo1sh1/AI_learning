import torch
import torch.nn.functional as F
from torchsummary import summary

#nn.relu == F.relu
class LeNet5(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 输入层：输入特征通道1、输出特征通道6，卷积核5、步长1，填充外界2单元
        self.conv1 = torch.nn.Conv2d(1, 6, 5,1,2)
        # 池化层：核2、步长2
        self.pool1 = torch.nn.AvgPool2d(2,2)

        self.conv2 = torch.nn.Conv2d(6, 16, 5,1)
        self.pool2 = torch.nn.AvgPool2d(2,2)
        # 全连接
        self.fc1 = torch.nn.Linear(16*5*5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
     x = self.pool1(F.relu(self.conv1(x)))
     x = self.pool2(F.relu(self.conv2(x)))
     # 展开
     x = x.view(-1, 16*5*5)
     x = F.relu(self.fc1(x))
     x = F.relu(self.fc2(x))
     x = self.fc3(x)
     return x