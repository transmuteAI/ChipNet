import math
import torch
import torch.nn as nn
from .layers import ModuleInjection, PrunableBatchNorm2d
from .base_model import BaseModel



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.activ = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        self.conv1, self.bn1 = ModuleInjection.make_prunable(self.conv1, self.bn1)
        self.conv2, self.bn2 = ModuleInjection.make_prunable(self.conv2, self.bn2)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activ(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.activ(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.activ = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.conv1, self.bn1 = ModuleInjection.make_prunable(self.conv1, self.bn1)
        self.conv2, self.bn2 = ModuleInjection.make_prunable(self.conv2, self.bn2)
        self.conv3, self.bn3 = ModuleInjection.make_prunable(self.conv3, self.bn3)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activ(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activ(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.activ(out)

        return out

class ResNetCifar(BaseModel):
    def __init__(self, block, layers, width=1, num_classes=1000):
        super(ResNetCifar, self).__init__()
        self.prunable_modules = []
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv1, self.bn1 = ModuleInjection.make_prunable(self.conv1, self.bn1)
        self.activ = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16 * width, layers[0])
        self.layer2 = self._make_layer(block, 32 * width, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64 * width, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(64 * width, num_classes)

        self.init_weights()

        assert block is BasicBlock
        for l_blocks in [self.layer1, self.layer2, self.layer3]:
            for b in l_blocks.children():
                downs = next(b.downsample.children()) if b.downsample is not None else None

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            conv_module = nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)
            bn_module = nn.BatchNorm2d(planes * block.expansion)
            conv_module, bn_module = ModuleInjection.make_prunable(conv_module, bn_module)
            if hasattr(bn_module, 'is_imp'):
                bn_module.is_imp = True
            downsample = nn.Sequential([conv_module, bn_module])

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activ(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

    def removable_orphans(self):
        num_removed = 0
        for l_blocks in [self.layer1, self.layer2, self.layer3]:
            for b in l_blocks:
                m1, m2 = b.bn1, b.bn2
                if self.is_all_pruned(m1) or self.is_all_pruned(m2):
                    num_removed += self.n_remaining(m1) + self.n_remaining(m2)
        return num_removed

    def remove_orphans(self):
        num_removed = 0
        for l_blocks in [self.layer1, self.layer2, self.layer3]:
            for b in l_blocks:
                m1, m2 = b.bn1, b.bn2
                if self.is_all_pruned(m1) or self.is_all_pruned(m2):
                    num_removed += self.n_remaining(m1) + self.n_remaining(m2)
                    m1.pruned_zeta.data.copy_(torch.zeros_like(m1.pruned_zeta))
                    m2.pruned_zeta.data.copy_(torch.zeros_like(m2.pruned_zeta))
        return num_removed
    
class ResNet(BaseModel):
    def __init__(self, block, layers, width=1, num_classes=1000, produce_vectors=False, init_weights=True):
        super(ResNet, self).__init__()
        self.prunable_modules = []
        self.frozen = []
        self.produce_vectors = produce_vectors
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1, self.bn1 = ModuleInjection.make_prunable(self.conv1, self.bn1)
        self.activ = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64 * width, layers[0])
        self.layer2 = self._make_layer(block, 128 * width, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256 * width, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512 * width, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)  # Global Avg Pool
        self.fc = nn.Linear(512 * block.expansion * width, num_classes)

        self.init_weights()

        assert block is Bottleneck
        for l in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for b in l.children():
                downs = next(b.downsample.children()) if b.downsample is not None else None

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            conv_module = nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)
            bn_module = nn.BatchNorm2d(planes * block.expansion)
            conv_module, bn_module = ModuleInjection.make_prunable(conv_module, bn_module)
            if hasattr(bn_module, 'is_imp'):
                bn_module.is_imp = True
            downsample = nn.Sequential([conv_module, bn_module])

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activ(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        feature_vectors = x.view(x.size(0), -1)
        x = self.fc(feature_vectors)

        if self.produce_vectors:
            return x, feature_vectors
        else:
            return x

    def removable_orphans(self):
        num_removed = 0
        for l_blocks in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for b in l_blocks:
                m1, m2, m3 = b.bn1, b.bn2, b.bn3
                if self.is_all_pruned(m1) or self.is_all_pruned(m2) or self.is_all_pruned(m3):
                    num_removed += self.n_remaining(m1) + self.n_remaining(m2) + self.n_remaining(m3)
        return num_removed

    def remove_orphans(self):
        num_removed = 0
        for l_blocks in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for b in l_blocks:
                m1, m2, m3 = b.bn1, b.bn2, b.bn3
                if self.is_all_pruned(m1) or self.is_all_pruned(m2) or self.is_all_pruned(m3):
                    num_removed += self.n_remaining(m1) + self.n_remaining(m2) + self.n_remaining(m3)
                    m1.pruned_zeta.data.copy_(torch.zeros_like(m1.pruned_zeta))
                    m2.pruned_zeta.data.copy_(torch.zeros_like(m2.pruned_zeta))
                    m3.pruned_zeta.data.copy_(torch.zeros_like(m3.pruned_zeta))
        return num_removed


def make_wide_resnet(num_classes):
    model = ResNetCifar(BasicBlock, [4, 4, 4], width=12, num_classes=num_classes)
    return model

def make_resnet32(num_classes):
    model = ResNetCifar(BasicBlock, [5, 5, 5], width=1, num_classes=num_classes)
    return model

def make_resnet164(num_classes):
    model = ResNetCifar(BasicBlock, [27, 27, 27], width=1, num_classes=num_classes)
    return model

def make_resnet50(num_classes):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    return model

def make_resnet101(num_classes):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)
    return model

def make_resnet152(num_classes):
    model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)
    return model

def get_resnet_model(model, method, num_classes):
    """Returns the requested model, ready for training/pruning with the specified method.

    :param model: str, either wrn or r50
    :param method: full or prune
    :param num_classes: int, num classes in the dataset
    :return: A prunable ResNet model
    """
    ModuleInjection.pruning_method = method
    if model == 'wrn':
        net = make_wide_resnet(num_classes)
    elif model == 'r32':
        net = make_resnet32(num_classes)
    elif model == 'r50':
        net = make_resnet50(num_classes)
    elif model == 'r101':
        net = make_resnet101(num_classes)
    elif model == 'r152':
        net = make_resnet152(num_classes)
    elif model == 'r164':
        net = make_resnet164(num_classes)
    net.prunable_modules = ModuleInjection.prunable_modules
    return net