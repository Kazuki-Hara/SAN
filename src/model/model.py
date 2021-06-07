import torch.nn as nn
from torch.nn.modules.module import Module


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=True)


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=1, stride=stride)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, first_block=False):
        super(Bottleneck, self).__init__()
        if in_channels != out_channels:
            self.shortcut = conv1x1(in_channels, out_channels)
            if first_block:
                stride = 1
            else:
                stride = 2
        else:
            stride = 1
        middle_dim = int(in_channels/4)
        self.conv1 = conv1x1(in_channels, middle_dim)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = conv3x3(middle_dim, middle_dim, stride=stride)
        self.bn2 = nn.BatchNorm2d(middle_dim)
        self.conv3 = conv1x1(middle_dim, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.Relu(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)
        if self.is_dim_changed:
            identity = self.shortcut(identity)
        out += identity
        out = self.relu(out)
        return out


class ResNet50(nn, Module):
    def __init__(self):
        super(ResNet50, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=7, padding=3, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.resblock1 = Bottleneck(64, 256, True)
        self.resblock2 = Bottleneck(256, 256)
        self.resblock3 = Bottleneck(256, 256)
        self.resblock4 = Bottleneck(256, 512)
        self.resblock5 = Bottleneck(512, 512)
        self.resblock6 = Bottleneck(512, 512)
        self.resblock7 = Bottleneck(512, 512)
        self.resblock8 = Bottleneck(512, 1024)
        self.resblock9 = Bottleneck(1024, 1024)
        self.resblock10 =Bottleneck(1024, 1024)
        self.resblock11 =Bottleneck(1024, 1024)
        self.resblock12 =Bottleneck(1024, 1024)
        self.resblock13 =Bottleneck(1024, 1024)
        self.resblock14 =Bottleneck(1024, 2048)
        self.resblock15 =Bottleneck(2048, 2048)
        self.resblock16 =Bottleneck(2048, 2048)

        self.glob_ave_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 100)

    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.resblock6(x)
        x = self.resblock7(x)
        x = self.resblock8(x)
        x = self.resblock9(x)
        x = self.resblock10(x)
        x = self.resblock11(x)
        x = self.resblock12(x)
        x = self.resblock13(x)
        x = self.resblock14(x)
        x = self.resblock15(x)
        x = self.resblock16(x)

        x = self.glob_ave_pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x