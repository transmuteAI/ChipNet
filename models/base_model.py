import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from .layers import PrunableBatchNorm2d

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.prunable_modules = []
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def n_remaining(self, m):
        return (m.get_binary_zetas()).sum()
    
    def is_all_pruned(self, m):
        return self.n_remaining(m) == 0

    def get_remaining(self, budget_type = 'channel_ratio'):
        """return the fraction of active zeta (i.e > 0)""" 
        n_rem = 0
        n_total = 0
        for l_block in self.prunable_modules:
            if budget_type == 'channel_ratio':
                n_rem += self.n_remaining(l_block)
                n_total += l_block.num_gates
            else:
                raise ValueError("Budget not defined!")
        return (n_rem-self.removable_orphans())/n_total

    def give_zetas(self):
        zetas = []
        for l_block in self.prunable_modules:
            zetas.append(l_block.zeta.cpu().detach().numpy().tolist())
        zetas = [z for k in zetas for z in k ]
        return zetas

    def get_crispnessLoss(self, device):
        """loss reponsible for making zeta 1 or 0"""
        loss = torch.FloatTensor([]).to(device)
        for l_block in self.prunable_modules:
            loss = torch.cat([loss, torch.pow(l_block.get_zeta_t()-l_block.get_zeta_i(), 2)])
        return torch.mean(loss).to(device)

    def prune(self, Vc, budget_type = 'channel_ratio', finetuning=False):
        """prunes the network to make zeta exactly 1 and 0"""
        if budget_type != 'channel_ratio':
            raise ValueError("Budget not defined!")

        for l_block in self.prunable_modules:
            l_block.prune()

        if finetuning:
            self.remove_orphans()
        else:
            problem = self.check_abnormality()
            return problem

    def unprune(self):
        for l_block in self.prunable_modules:
            l_block.unprune()

    def freeze_weights(self):
        self.requires_grad = False
        for l_block in self.prunable_modules:
            l_block.unprune
    
    def prepare_for_finetuning(self, device, budget, budget_type = 'channel_ratio'):
        """freezes zeta"""
        if budget_type != 'channel_ratio':
            ValueError("Budget not defined!")
        self.device = device
        self(torch.rand(2,3,32,32).to(device))
        self.prune(budget, budget_type=budget_type, finetuning=True)     

    def get_channels(self):
        total_channels = 0.
        active_channels = 0.
        for l_block in self.prunable_modules:
                active_channels+=l_block.pruned_zeta.sum().item()
                total_channels+=l_block.num_gates
        return active_channels, total_channels

    def set_beta_gamma(self, beta, gamma):
        for l_block in self.prunable_modules:
            l_block.set_beta_gamma(beta, gamma)

    def check_abnormality(self):
        n_removable = self.removable_orphans()
        isbroken = self.check_if_broken()
        if n_removable!=0. and isbroken:
            return f'both rem_{n_removable} and broken'
        if n_removable!=0.:
            return f'removable_{n_removable}'
        if isbroken:
            return 'broken'

    def check_if_broken(self):
        for bn in self.prunable_modules:
            if bn.is_imp and bn.pruned_zeta.sum()==0:
                return True
        return False