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
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2, self.bn2 = ModuleInjection.make_prunable(self.conv2, self.bn2)
        self.bn2.is_imp = True

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNetv1(BaseModel):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=10):
        super(MobileNetv1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv1, self.bn1 = ModuleInjection.make_prunable(self.conv1, self.bn1)
        self.bn1.is_imp = True
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(1024, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def removable_orphans(self):
        num_removed = 0
        for l_blocks in self.layers:
            for b in l_blocks:
                m2 = b.bn2
                if self.is_all_pruned(m2):
                    num_removed += self.n_remaining(m2)
        return num_removed

    def remove_orphans(self):
        return 0



def get_mobilenet(model, method, num_classes):
    """Returns the requested model, ready for training/pruning with the specified method.

    :param model: str
    :param method: full or prune
    :param num_classes: int, num classes in the dataset
    :return: A prunable MobileNet model
    """
    ModuleInjection.pruning_method = method
    ModuleInjection.prunable_modules = []
    if model == 'mobilenetv1':
        net = MobileNetv1(num_classes)
    net.prunable_modules = ModuleInjection.prunable_modules
    return net
