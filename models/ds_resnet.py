'''
ResNet with Double skip connections
Under construction!! 
'''
import torch
import torch.nn as nn
from .layers import ModuleInjection, PrunableBatchNorm2d
from .base_model import BaseModel
import numpy as np


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
        
        if self.downsample is not None : 
            residual = self.downsample(x)
            
        out += residual
        out = self.activ(out)
        residual = out
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.activ(out)

        return out

# class Bottleneck(nn.Module):
#     expansion = 4
#     name = "Bottleneck"

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * self.expansion)
#         self.activ = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride

#         self.conv1, self.bn1 = ModuleInjection.make_prunable(self.conv1, self.bn1)
#         self.conv2, self.bn2 = ModuleInjection.make_prunable(self.conv2, self.bn2)
#         self.conv3, self.bn3 = ModuleInjection.make_prunable(self.conv3, self.bn3)

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.activ(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.activ(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.activ(out)

#         return out

class ResNet(BaseModel):
    def __init__(self, block, layers, width=1, num_classes=1000, produce_vectors=False, init_weights=True, insize=32):
        super(ResNet, self).__init__()
        self.insize=insize
        self.width = width
        self.produce_vectors = produce_vectors
        self.block_type = block.__class__.__name__
        self.inplanes = 64
        self.layers_size = layers
        if insize<128:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1, self.bn1 = ModuleInjection.make_prunable(self.conv1, self.bn1)
        self.prev_module[self.bn1]=None
        self.activ = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64 * width, layers[0])
        self.layer2 = self._make_layer(block, 128 * width, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256 * width, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512 * width, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)  # Global Avg Pool
        self.fc = nn.Linear(512 * block.expansion * width, num_classes)

        self.init_weights()

        if block.name =="BasicBlock":
            prev = self.bn1
            for l_block in [self.layer1 , self.layer2 , self.layer3 , self.layer4]:
                for b in l_block:
                    self.prev_module[b.bn1] = prev
                    if b.downsample is not None:
                        self.prev_module[b.downsample[1]] = prev
                        prev = (b.downsample[1] , b.bn1)
                    else:
                        prev = (prev ,b.bn1)
                    self.prev_module[b.bn2] = prev
                    prev = (prev , b.bn2)
                    
        else:
            raise ValueError("Only BasicBlock supported")        
        #     prev = self.bn1
        #     for l_block in [self.layer1, self.layer2, self.layer3, self.layer4]:
        #         for b in l_block:
        #             self.prev_module[b.bn1] = prev
        #             self.prev_module[b.bn2] = b.bn1
        #             self.prev_module[b.bn3] = b.bn2
        #             if b.downsample is not None:
        #                 self.prev_module[b.downsample[1]] = prev
        #                 prev = (b.downsample[1], b.bn3)
        #             else:
        #                 prev = (prev, b.bn3)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            conv_module = nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)
            bn_module = nn.BatchNorm2d(planes * block.expansion)
            conv_module, bn_module = ModuleInjection.make_prunable(conv_module, bn_module)
            if hasattr(bn_module, 'is_imp'):
                bn_module.is_imp = True
            downsample = nn.Sequential(conv_module, bn_module)

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
        return 0

    def remove_orphans(self):
        return 0

    def __calc_params(self, a):
        ans = a[0]*a[1]*9
        current_loc = 2
        current_max = a[1]
        downsample_n = a[2]
        do_downsample = True if self.width>1 else False
        for l in self.layers_size:
            for i in range(l):
                if do_downsample:
                    downsample_n = a[current_loc]
                    ans+=current_max*a[current_loc]
                    current_loc+=1
            
                ans+=current_max*a[current_loc]*9
                ans+=a[current_loc]*a[current_loc+1]*9
                if do_downsample:
                    current_max = max(downsample_n, a[current_loc+1])
                else:
                    current_max = max(current_max, a[current_loc+1])
                do_downsample = False
                current_loc+=2
            do_downsample = True
        return ans + a[-1]*self.num_classes + 2*np.sum(a)

    def params(self):
        a = [3]
        b = [3]
        for i in self.prunable_modules:
            a.append(int(i.pruned_zeta.sum()))
            b.append(len(i.pruned_zeta))
        return self.__calc_params(a)/self.__calc_params(b)

def make_resnet18(num_classes, insize):
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, insize=insize)
    return model

# def make_resnet50(num_classes, insize):
#     model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, insize=insize)
#     return model

# def make_resnet101(num_classes, insize):
#     model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, insize=insize)
#     return model

# def make_resnet152(num_classes, insize):
#     model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, insize=insize)
#     return model

def get_ds_resnet_model(model, method, num_classes, insize):
    """Returns the requested model, ready for training/pruning with the specified method.

    :param model: str, either wrn or r50
    :param method: full or prune
    :param num_classes: int, num classes in the dataset
    :return: A prunable ResNet model
    """
    ModuleInjection.pruning_method = method
    ModuleInjection.prunable_modules = []

    if model == 'dsr18':
        net = make_resnet18(num_classes, insize)
    # elif model == 'ds50':
    #     net = make_resnet50(num_classes, insize)
    # elif model == 'dsr101':
    #     net = make_resnet101(num_classes, insize)
    # elif model == 'dsr152':
    #     net = make_resnet152(num_classes, insize)
    net.prunable_modules = ModuleInjection.prunable_modules
    return net