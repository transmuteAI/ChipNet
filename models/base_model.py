import torch
import torch.nn as nn
import numpy as np
from .layers import PrunableBatchNorm2d

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.prunable_modules = []
        pass
    
    def set_threshold(self, threshold):
        self.prune_threshold = threshold
        
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

    def calculate_prune_threshold(self, Vc, budget_type = 'channel_ratio'):
        zetas = self.give_zetas()
        if budget_type == 'volume_ratio':
            zeta_weights = self.give_zeta_weights()
            zeta_weights = zeta_weights[np.argsort(zetas)]
        zetas = sorted(zetas)
        if budget_type == 'volume_ratio':
            curr_budget = 0
            indx = 0
            while(curr_budget<Vc):
                indx+=1
                curr_budget+=zeta_weights[indx]
            prune_threshold = zetas[indx]
        else:
            prune_threshold = zetas[int((1.-Vc)*len(zetas))]
        return prune_threshold
    
    def smoothRound(self, x, steepness=20.):
        return 1./(1.+torch.exp(-1*steepness*(x-0.5)))
    
    def n_remaining(self, m, steepness=20.):
        return (m.pruned_zeta if m.is_pruned else self.smoothRound(m.get_zeta_t(), steepness)).sum()
    
    def is_all_pruned(self, m):
        return self.n_remaining(m) == 0
    
    def get_remaining(self, steepness=20., budget_type = 'channel_ratio'):
        """return the fraction of active zeta_t (i.e > 0.5)""" 
        n_rem = 0
        n_total = 0
        for l_block in self.prunable_modules:
            if budget_type == 'volume_ratio':
                n_rem += (self.n_remaining(l_block, steepness)*l_block._conv_module.output_volume)
                n_total += (l_block.num_gates*l_block._conv_module.output_volume)
            elif budget_type == 'channel_ratio':
                n_rem += self.n_remaining(l_block, steepness)
                n_total += l_block.num_gates
        return n_rem/n_total

    def give_zetas(self):
        zetas = []
        for l_block in self.prunable_modules:
            zetas.append(l_block.get_zeta_t().cpu().detach().numpy().tolist())
        zetas = [z for k in zetas for z in k ]
        return zetas

    def give_zeta_weights(self):
        zeta_weights = []
        for l_block in self.prunable_modules:
            zeta_weights.append([l_block._conv_module.output_volume]*l_block.num_gates)
        zeta_weights = [z for k in zeta_weights for z in k ]
        return zeta_weights/np.sum(zeta_weights)

    def plot_zt(self):
        """plots the distribution of zeta_t and returns the same"""
        zetas = self.give_zetas()
        exactly_zeros = np.sum(np.array(zetas)==0.0)
        exactly_ones = np.sum(np.array(zetas)==1.0)
        plt.hist(zetas)
        plt.show()
        return exactly_zeros, exactly_ones
    
    def get_crispnessLoss(self, device):
        """loss reponsible for making zeta_t 1 or 0"""
        loss = torch.FloatTensor([]).to(device)
        for l_block in self.modules():
            if isinstance(l_block, PrunableBatchNorm2d):
                loss = torch.cat([loss, torch.pow(l_block.get_zeta_t()-l_block.get_zeta_i(), 2)])
        return torch.mean(loss).to(device)

    def prune(self, Vc, budget_type = 'channel_ratio', finetuning=False, threshold=None):
        """prunes the network to make zeta_t exactly 1 and 0"""
        if threshold==None:
            self.prune_threshold = self.calculate_prune_threshold(Vc, budget_type)
            threshold = min(self.prune_threshold, 0.9)
            
        for l_block in self.modules():
            if isinstance(l_block, PrunableBatchNorm2d):
                l_block.prune(threshold)

        if finetuning:
            self.remove_orphans()
            return threshold
        else:
            problem = self.check_abnormality()
            return threshold, problem

    def unprune(self):
        for l_block in self.modules():
            if isinstance(l_block, PrunableBatchNorm2d):
                l_block.unprune()
    
    def prepare_for_finetuning(self, device, budget, budget_type = 'channel_ratio'):
        """freezes zeta"""
        self.device = device
        threshold = self.prune(budget, budget_type=budget_type, finetuning=True)
        while self.get_remaining()<budget:
            threshold-=0.0001
            self.prune(budget, finetuning=True, budget_type=budget_type, threshold=threshold)

        return threshold      

    def get_params_count(self):
        total_params = 0
        active_params = 0
        for l_block in self.modules():
            if isinstance(l_block, PrunableBatchNorm2d):
                active_param, total_param = l_block.get_params_count()
                active_params+=active_param 
                total_params+=total_param
            if isinstance(l_block, nn.Linear):
                linear_params = l_block.weight.view(-1).shape[0]
                active_params+=linear_params
                total_params+=linear_params
        return active_params, total_params

    def get_volume(self):
        total_volume = 0.
        active_volume = 0.
        for l_block in self.prunable_modules:
                active_volume_, total_volume_ = l_block.get_volume()
                active_volume+=active_volume_ 
                total_volume+=total_volume_
        return active_volume, total_volume
          
    def set_beta_gamma(self, beta, gamma):
        for l_block in self.modules():
            if isinstance(l_block, PrunableBatchNorm2d):
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
        for bn in self.modules():
            if isinstance(bn, PrunableBatchNorm2d) and bn.is_imp:
                if bn.pruned_zeta.sum()==0:
                    return True
        return False
