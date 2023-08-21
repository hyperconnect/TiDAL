'''ResNet & VGG in PyTorch.

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
import math


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BasicBlock2(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock2, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.dropout(F.relu(self.bn1(self.conv1(x))), p=0.3, training=True)
        out = F.dropout(self.bn2(self.conv2(out)), p=0.3, training=True)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        # self.linear2 = nn.Linear(1000, num_classes)

        self.dropout = nn.Dropout(0.5)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, method=None):
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = F.avg_pool2d(out4, 4)
        outf = out.view(out.size(0), -1)
        # outl = self.linear(outf)

        if method == 'BALD':
            outf = self.dropout(outf)

        out = self.linear(outf)
        return out, outf, [out1, out2, out3, out4]


class ResNetfm(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetfm, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        # self.linear2 = nn.Linear(1000, num_classes)
        self.dropout = nn.Dropout(0.2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, method=None):
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = F.avg_pool2d(out4, 4)
        outf = out.view(out.size(0), -1)
        # outl = self.linear(outf)

        if method == 'BALD':
            outf = self.dropout(outf)

        out = self.linear(outf)
        return out, outf, [out1, out2, out3, out4]


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def ResNet18fm(num_classes=10):
    return ResNetfm(BasicBlock, [2, 2, 2, 2], num_classes)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


class ResNet34(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        import torchvision
        import os

        os.environ['TORCH_HOME'] = 'cache'  # hacky workaround to set model dir
        self.resnet34 = torchvision.models.resnet34(pretrained=True)
        self.resnet34.fc = nn.Identity()  # remote last fc
        # self.fc = nn.Linear(2048, num_classes)
        self.fc = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.fc.weight)
        self.fc.bias.data.zero_()

    def forward(self, x, method=None):  # (bs, C, H, W)
        x = self.resnet34.conv1(x)
        x = self.resnet34.bn1(x)
        x = self.resnet34.relu(x)
        out = self.resnet34.maxpool(x)

        out1 = self.resnet34.layer1(out)
        out2 = self.resnet34.layer2(out1)
        out3 = self.resnet34.layer3(out2)
        out4 = self.resnet34.layer4(out3)

        x = self.resnet34.avgpool(out4)
        outf = torch.flatten(x, 1)

        if method == 'BALD':
            outf = self.dropout(outf)

        out = self.fc(outf)
        return out, outf, [out1, out2, out3, out4]


class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        import torchvision
        import os

        os.environ['TORCH_HOME'] = 'cache'  # hacky workaround to set model dir
        self.resnet50 = torchvision.models.resnet50(pretrained=True)
        self.resnet50.fc = nn.Identity()  # remote last fc
        self.fc = nn.Linear(2048, num_classes)
        self.dropout = nn.Dropout(0.5)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.fc.weight)
        self.fc.bias.data.zero_()

    def forward(self, x, method=None):  # (bs, C, H, W)
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        out = self.resnet50.maxpool(x)

        out1 = self.resnet50.layer1(out)
        out2 = self.resnet50.layer2(out1)
        out3 = self.resnet50.layer3(out2)
        out4 = self.resnet50.layer4(out3)

        x = self.resnet50.avgpool(out4)
        outf = torch.flatten(x, 1)

        if method == 'BALD':
            outf = self.dropout(outf)

        out = self.fc(outf)
        return out, outf, [out1, out2, out3, out4]


class ResNet101(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        import torchvision
        import os

        os.environ['TORCH_HOME'] = 'cache'  # hacky workaround to set model dir
        self.resnet101 = torchvision.models.resnet101(pretrained=True)
        self.resnet101.fc = nn.Identity()  # remote last fc
        self.fc = nn.Linear(2048, num_classes)
        self.dropout = nn.Dropout(0.5)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.fc.weight)
        self.fc.bias.data.zero_()

    def forward(self, x, method=None):  # (bs, C, H, W)
        x = self.resnet101.conv1(x)
        x = self.resnet101.bn1(x)
        x = self.resnet101.relu(x)
        out = self.resnet101.maxpool(x)

        out1 = self.resnet101.layer1(out)
        out2 = self.resnet101.layer2(out1)
        out3 = self.resnet101.layer3(out2)
        out4 = self.resnet101.layer4(out3)

        x = self.resnet101.avgpool(out4)
        outf = torch.flatten(x, 1)

        if method == 'BALD':
            outf = self.dropout(outf)

        out = self.fc(outf)
        return out, outf, [out1, out2, out3, out4]


class VGG(nn.Module):
    '''
    VGG model 
    '''

    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        feat = []
        y = x
        for i, model in enumerate(self.features):
            y = model(y)
            if i in {3, 5, 10, 15}:
                feat.append(y)  # (y.view(y.size(0),-1))

        x = self.features(x)
        out4 = x.view(x.size(0), -1)
        x = self.classifier(out4)
        return x, out4, [feat[0], feat[1], feat[2], feat[3]]


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))
