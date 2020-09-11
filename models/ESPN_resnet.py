from __future__ import absolute_import

""" 
Code taken and modified from https://github.com/chomd90/extreme_sparse/blob/master/archs/cifar_resnet.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.init as init
from .layers import ModuleInjection, PrunableBatchNorm2d
from .base_model import BaseModel

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv1, self.bn1 = ModuleInjection.make_prunable(self.conv1, self.bn1)
        self.conv2, self.bn2 = ModuleInjection.make_prunable(self.conv2, self.bn2)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                conv = nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                bn = nn.BatchNorm2d(self.expansion * planes)
                conv, bn = ModuleInjection.make_prunable(conv, bn)
                if hasattr(bn, 'is_imp'):
                    bn.is_imp = True
                self.shortcut = nn.Sequential(conv, bn)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(BaseModel):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        
        _outputs = [32, 64, 128]
        self.in_planes = _outputs[0]
        self.conv1 = nn.Conv2d(3, _outputs[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(_outputs[0])
        self.conv1, self.bn1 = ModuleInjection.make_prunable(self.conv1, self.bn1)

        self.layer1 = self._make_layer(block, _outputs[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, _outputs[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, _outputs[2], num_blocks[2], stride=2)
        self.linear = nn.Linear(_outputs[2], num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = F.log_softmax(out, dim=1)
        return out

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

def resnet32(num_classes):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes)

def get_ESPN_model(method, num_classes):
    ModuleInjection.pruning_method = method
    net = resnet32(num_classes)
    net.prunable_modules = ModuleInjection.prunable_modules
    return net