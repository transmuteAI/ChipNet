from collections import defaultdict, OrderedDict
import math

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import numpy as np
import random 
import os
import matplotlib.pyplot as plt
from copy import deepcopy
from .layers import ModuleInjection, SparseConfig, PrunableBatchNorm2d




def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SuperModel(nn.Module):
    def __init__(self):
        super(SuperModel, self).__init__()
        pass
    
    def set_thresh(self, thresh):
        self.prune_thresh = thresh
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def cal_prune_thresh(self,Vc):
        zetas = self.give_zetas()
        zetas = sorted(zetas)
        prune_thresh = zetas[int((1.-Vc)*len(zetas))]
        return prune_thresh
    
    def __logistic(self, x, steepness = 20):
        return 1./(1.+torch.exp(-1*steepness*(x-0.5)))

    def get_remaining(self, steepness = 20.):
        """return the fraction of active zeta_t (i.e > 0.5)"""
        def n_remaining(m):
            return (self.__logistic(m.get_zeta_t(),steepness)).sum()  
        n_rem = 0
        n_total = 0
        for l_block in self.modules():
            if  isinstance(l_block, PrunableBatchNorm2d):
              n_rem += n_remaining(l_block)
              n_total += l_block.num_gates
        return n_rem/n_total

    def give_zetas(self):
        zetas = []
        for l_block in self.modules():
            if  isinstance(l_block, PrunableBatchNorm2d):
                zetas.append(l_block.get_zeta_t().cpu().detach().numpy().tolist())
        zetas = [z for k in zetas for z in k ]
        return zetas

    def plot_zt(self):
        """plots the distribution of zeta_t and returns the same"""
        zetas = self.give_zetas()
        exactly_zeros = np.sum(np.array(zetas)==0.0)
        exactly_ones = np.sum(np.array(zetas)==1.0)
        print("Num of exactly zeros:",exactly_zeros)
        print("Num of exactly one:",exactly_ones)
        plt.hist(zetas)
        plt.show()
        zetas = sorted(zetas)
        return exactly_zeros, exactly_ones
    
    def get_zt_diff_zi_sq_loss(self,device):
        """loss reponsible for making zeta_t 1 or 0"""
        loss = torch.FloatTensor([]).to(device)
        for l_block in self.modules():
            if  isinstance(l_block, PrunableBatchNorm2d):
              loss = torch.cat([loss,torch.pow(l_block.get_zeta_t()-l_block.get_zeta_i(), 2)])
        return torch.mean(loss)
    
    def get_bn_sq_loss(self,device):
        """loss for regularizing batchnorm"""
        loss = torch.FloatTensor([]).to(device)
        for l_block in self.modules():
            if  isinstance(l_block, PrunableBatchNorm2d):
              loss = torch.cat([loss,torch.pow(l_block.weight, 2)])
              loss = torch.cat([loss,torch.pow(l_block.bias, 2)])
        return torch.mean(loss)
        
        
    def prune_net(self, Vc ,finetuning = False, thresh=None):
        """prunes the network to make zeta_t exactly 1 and 0"""
        if(thresh==None):
            self.prune_thresh = self.cal_prune_thresh(Vc)
            thresh = min(self.prune_thresh,0.9)
        print(f"PRUNING_THRESH USED: {thresh}")
        for l_block in self.modules():
            if  isinstance(l_block, PrunableBatchNorm2d):
              l_block.zeta.data[l_block.get_zeta_t().cpu().detach().numpy()>=thresh]=1000000.
              l_block.zeta.data[l_block.get_zeta_t().cpu().detach().numpy()<thresh]=-1000000.
        if(finetuning):
            self.remove_orphans()
            return thresh
        else:
            problem = self.check_abnormality()
            return thresh, problem

    def freeze_bn(self):
        """freezes the batchnorm"""
        for l_block in self.modules():
            if  isinstance(l_block, PrunableBatchNorm2d):
              l_block.weight.requires_grad = False
              l_block.bias.requires_grad = False
                
    def freeze_except_zeta(self):
        """freezes whole network except zeta and classifier"""
        for l_block in self.modules():
            if(isinstance(l_block, nn.Conv2d)):
                l_block.weight.requires_grad = False
            elif(isinstance(l_block, PrunableBatchNorm2d)):
                l_block.bias.requires_grad = False
                l_block.weight.requires_grad = False
                l_block.zeta.requires_grad = True
            elif(isinstance(l_block, nn.Linear)):
                l_block.bias.requires_grad = True
                l_block.weight.requires_grad = True
                
    def unfreeze_except_zeta(self,beta,gamma,device,Vc, finetuning = False):
        """freezes zeta"""
        self.device = device
        self.set_beta_gamma(beta,gamma,device)
        if(finetuning):
            d = deepcopy(self)
            thresh = d.prune_net(Vc, finetuning)
            while(d.get_remaining(60.)<Vc):
                thresh-=0.0001
                d = deepcopy(self)
                d.prune_net(Vc, finetuning,thresh)
            self.prune_net(Vc, finetuning,thresh)
            del d
        else:
            self.prune_net(Vc, finetuning)
        for l_block in self.modules():
            if(isinstance(l_block, nn.Conv2d)):
                l_block.weight.requires_grad = True
            if  isinstance(l_block, PrunableBatchNorm2d):
                l_block.bias.requires_grad = True
                l_block.weight.requires_grad = True
                l_block.zeta.requires_grad = False
            if(isinstance(l_block, nn.Linear)):
                l_block.bias.requires_grad = True
                l_block.weight.requires_grad = True
                
    def isfreezed(self):
        """sanity check to ensure freezing"""
        for l_block in self.modules():
            if(isinstance(l_block, nn.Conv2d)):
                print(l_block.weight.requires_grad)
            elif(isinstance(l_block, PrunableBatchNorm2d)):
                print(l_block.bias.requires_grad) 
                print(l_block.weight.requires_grad) 
                
          
    def set_beta_gamma(self, beta, gamma, device):
        for l_block in self.modules():
            if  isinstance(l_block, PrunableBatchNorm2d):
                l_block.set_beta_gamma(beta, gamma, device)
    
    def get_grads(self):
        zeta_grads = []
        for l_block in self.modules():
            if  isinstance(l_block, PrunableBatchNorm2d):
                zeta_grads.append(l_block.get_grads())
        zeta_grads = [z for k in zeta_grads for z in k ]
        return sorted(zeta_grads)
    
    def check_abnormality(self):
        n_removable = self.removable_orphans()
        isbroken = self.check_if_broken()
        if(n_removable!=0. and isbroken):
            return f'both rem_{n_removable}'
        if(n_removable!=0.):
            return f'removable_{n_removable}'
        if(isbroken):
            return 'broken'
        
    def check_if_broken(self):
        def n_remaining(m):
            return (m.get_zeta_t()==1.).sum()
        for bn in self.modules():
            if (isinstance(bn, PrunableBatchNorm2d) and bn.isimp):
                if(n_remaining(m)==0):
                    return True
        return False       
    

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

class ResNetCifar(SuperModel):
    def __init__(self, block, layers, width=1, num_classes=1000):
        self.prunable_modules = []
        self.inplanes = 16
        super(ResNetCifar, self).__init__()
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
            if hasattr(bn_module, 'isimp'):
                bn_module.isimp = True
            downsample = nn.Sequential(*ModuleInjection.make_prunable(conv_module, bn_module))

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

        def n_remaining(m):
            return (m.get_zeta_t()==1.).sum()

        def is_all_pruned(m):
            return (n_remaining(m)) == 0
        num_removed = 0
        for l_blocks in [self.layer1, self.layer2, self.layer3]:
            for b in l_blocks:
                m1, m2 = b.bn1, b.bn2
                if is_all_pruned(m1) or is_all_pruned(m2):
                    num_removed += n_remaining(m1) + n_remaining(m2)
        return num_removed

    def remove_orphans(self):

        def n_remaining(m):
            return (m.get_zeta_t()==1.).sum()

        def is_all_pruned(m):
            return (n_remaining(m)) == 0
        num_removed = 0
        for l_blocks in [self.layer1, self.layer2, self.layer3]:
            for b in l_blocks:
                m1, m2 = b.bn1, b.bn2
                if is_all_pruned(m1) or is_all_pruned(m2):
                    num_removed += n_remaining(m1) + n_remaining(m2)
                    m1.zeta.data = (torch.ones_like(m1.zeta)*(-100000)).to(self.device)
                    m2.zeta.data = (torch.ones_like(m2.zeta)*(-100000)).to(self.device)
        return num_removed
    
class ResNet(SuperModel):
    def __init__(self, block, layers, width=1, num_classes=1000, produce_vectors=False, init_weights=True):
        self.prunable_modules = []
        self.frozen = []
        self.produce_vectors = produce_vectors
        self.inplanes = 64
        super(ResNet, self).__init__()
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


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            conv_module = nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)
            bn_module = nn.BatchNorm2d(planes * block.expansion)
            if hasattr(bn_module, 'isimp'):
                bn_module.isimp = True
            downsample = nn.Sequential(*ModuleInjection.make_prunable(conv_module, bn_module))

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

        def n_remaining(m):
            return (m.get_zeta_t()==1.).sum()

        def is_all_pruned(m):
            return (n_remaining(m)) == 0
        num_removed = 0
        for l_blocks in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for b in l_blocks:
                m1, m2, m3 = b.bn1, b.bn2, b.bn3
                if is_all_pruned(m1) or is_all_pruned(m2) or is_all_pruned(m3):
                    num_removed += n_remaining(m1) + n_remaining(m2) + n_remaining(m3)
        return num_removed

    def remove_orphans(self):

        def n_remaining(m):
            return (m.get_zeta_t()==1.).sum()

        def is_all_pruned(m):
            return (n_remaining(m)) == 0
        num_removed = 0
        for l_blocks in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for b in l_blocks:
                m1, m2, m3 = b.bn1, b.bn2, b.bn3
                if is_all_pruned(m1) or is_all_pruned(m2) or is_all_pruned(m3):
                    num_removed += n_remaining(m1) + n_remaining(m2) + n_remaining(m3)
                    m1.zeta.data = (torch.ones_like(m1.zeta)*(-100000)).to(self.device)
                    m2.zeta.data = (torch.ones_like(m2.zeta)*(-100000)).to(self.device)
                    m3.zeta.data = (torch.ones_like(m3.zeta)*(-100000)).to(self.device)
        return num_removed

def make_wide_resnet(num_classes):
    model = ResNetCifar(BasicBlock, [4, 4, 4], width=12, num_classes=num_classes)
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

def get_model(model, method, num_classes):
    """Returns the requested model, ready for training/pruning with the specified method.

    :param model: str, either wrn or r50
    :param method: full or prune
    :param num_classes: int, num classes in the dataset
    :return: A prunable ResNet model
    """
    ModuleInjection.pruning_method = method

    if model == 'wrn':
        net = make_wide_resnet(num_classes)
    if model == 'r50':
        net = make_resnet50(num_classes)
    if model == 'r101':
        net = make_resnet101(num_classes)
    if model == 'r152':
        net = make_resnet152(num_classes)
    net.prunable_modules = ModuleInjection.prunable_modules
    return net