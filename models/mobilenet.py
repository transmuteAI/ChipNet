import math
import torch
import torch.nn as nn
from .layers import ModuleInjection, PrunableBatchNorm2d
from .base_model import BaseModel
import numpy as np

'''MobileNet in PyTorch.
See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
Code is taken from https://github.com/kuangliu/pytorch-cifar/blob/master/models/mobilenet.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv1, self.bn1 = ModuleInjection.make_prunable(self.conv1, self.bn1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.conv3, self.bn3 = ModuleInjection.make_prunable(self.conv3, self.bn3)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            conv_module = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
            bn_module = nn.BatchNorm2d(out_planes)
            conv_module, bn_module = ModuleInjection.make_prunable(conv_module, bn_module)
            if hasattr(bn_module, 'is_imp'):
                bn_module.is_imp = True
            self.shortcut = nn.Sequential(
                conv_module,
                bn_module
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetv2(BaseModel):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetv2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv1, self.bn1 = ModuleInjection.make_prunable(self.conv1, self.bn1)
        if hasattr(self.bn1, 'is_imp'):
            self.bn1.is_imp = True
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.conv2, self.bn2 = ModuleInjection.make_prunable(self.conv2, self.bn2)
        if hasattr(self.bn2, 'is_imp'):
            self.bn2.is_imp = True
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def removable_orphans(self):
        num_removed = 0
        for b in self.layers:
            m1, m2 = b.bn1, b.bn3
            if self.is_all_pruned(m1) or self.is_all_pruned(m2):
                num_removed += self.n_remaining(m1) + self.n_remaining(m2)
        return num_removed

    def remove_orphans(self):
        num_removed = 0
        for b in self.layers:
            m1, m2 = b.bn1, b.bn3
            if self.is_all_pruned(m1) or self.is_all_pruned(m2):
                num_removed += self.n_remaining(m1) + self.n_remaining(m2)
                m1.pruned_zeta.data.copy_(torch.zeros_like(m1.pruned_zeta))
                m2.pruned_zeta.data.copy_(torch.zeros_like(m2.pruned_zeta))
        return num_removed



def get_mobilenet(model, method, num_classes):
    """Returns the requested model, ready for training/pruning with the specified method.

    :param model: str
    :param method: full or prune
    :param num_classes: int, num classes in the dataset
    :return: A prunable MobileNet model
    """
    ModuleInjection.pruning_method = method
    ModuleInjection.prunable_modules = []
    if model == 'mobilenetv2':
        net = MobileNetv2(num_classes)
    net.prunable_modules = ModuleInjection.prunable_modules
    return net
