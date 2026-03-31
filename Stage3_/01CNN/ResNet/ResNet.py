import torch
import torch.nn as nn
from torch import Tensor
from torchsummary import summary

# 基础
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.aa = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
        )

        # 是否虚线shortcut
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.aa(x)

        # 如果存在则先进行downsample(一个打包好的Sequential对象)
        if self.downsample is not None:
            identity = self.downsample(x)

        # 加和区(存在加if线，不存在直接相加不需要变化)
        out += identity
        out = self.relu(out)

        return out


# 带漏斗结构
class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion: int = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1):
        super().__init__()
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()

        self.inplanes = 64
        # self.dilation = 1
        # 此处为接受图像的第一层卷积，输入频道3，输出64频道，核大小7，步长2，填充same但是计算为3，不加入偏置因为BN层会自动剔除均值把b消除所以b在这里没用
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 以上为固定开始

        # 通过输入layers不同，初始化不同的resnet网络结构，但总体都为4块，且通道64→128→256→512
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # 什么情况会引入shortcut捷径分支虚线 （步长不为1，输入通道不等于(上一步)输出通道的n倍）
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 先使用1大小的核进行通道数转化
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # 第一个元素有实线有虚线，所以用block带downsample初始化第一个元素(自动计算实线虚线)
        layers.append(
            block(self.inplanes, planes, stride, downsample)
        )
        # 第一次初始化可能会改变下一步的输入通道数
        # block下的expansion属性为通道变化倍率
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            # 继续初始化其他块
            layers.append(
                block(self.inplanes, planes)
            )
        # 输出整个打包好的.Sequential对象
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18(num_classes=1000):
    # https://download.pytorch.org/models/resnet18-f37072fd.pth
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def resnet34(num_classes=1000):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def resnet50(num_classes=1000):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


if __name__ == '__main__':
    # model = resnet18(num_classes=3).to("cuda")
    # model = resnet34(num_classes=3).to("cuda")
    model = resnet50(num_classes=3).to("cuda")
    fc_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(fc_inputs, 3),
        # nn.ReLU(),
        # nn.Dropout(0.4),
        # nn.Linear(256, 3),
    ).to("cuda")
    print(summary(model, (3, 224, 224)))