import math
import torch
import torch.nn as nn
from .layers import ModuleInjection, PrunableBatchNorm2d
from .base_model import BaseModel


class VGG(BaseModel):

    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self.init_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def remove_orphans(self):
        return 0
    
    def removable_orphans(self):
        return 0

def make_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            bn = nn.BatchNorm2d(v)
            conv2d, bn = ModuleInjection.make_prunable(conv2d, bn)
            if hasattr(bn_module, 'is_imp'):
                bn.is_imp = True
            layers += [conv2d, bn, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def _vgg(arch, cfg, num_classes):
    model = VGG(make_layers(cfgs[cfg]), num_classes=num_classes)
    return model

def make_vgg11_bn(num_classes):
    return _vgg('vgg11_bn', 'A', num_classes)

def make_vgg13_bn(num_classes):
    return _vgg('vgg13_bn', 'B', num_classes)

def make_vgg16_bn(num_classes):
    return _vgg('vgg16_bn', 'D', num_classes)

def make_vgg19_bn(num_classes):
    return _vgg('vgg19_bn', 'E', num_classes)


def get_vgg_model(model, method, num_classes):
    """Returns the requested model, ready for training/pruning with the specified method.

    :param model: str, either wrn or r50
    :param method: full or prune
    :param num_classes: int, num classes in the dataset
    :return: A prunable ResNet model
    """
    ModuleInjection.pruning_method = method
    if model == 'VGG11':
        net = make_vgg11_bn(num_classes)
    elif model == 'VGG13':
        net = make_vgg13_bn(num_classes)
    elif model == 'VGG16':
        net = make_vgg16_bn(num_classes)
    elif model == 'VGG19':
        net = make_vgg19_bn(num_classes)
    net.prunable_modules = ModuleInjection.prunable_modules
    return net