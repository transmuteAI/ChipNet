import torch
import torch.nn as nn
import numpy as np

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        pass
    
    def set_threshold(self, threshold):
        self.prune_threshold = threshold
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def calculate_prune_threshold(self, Vc):
        zetas = self.give_zetas()
        zetas = sorted(zetas)
        prune_threshold = zetas[int((1.-Vc)*len(zetas))]
        return prune_threshold
    
    def smoothRound(self, x, steepness=20.):
        return 1./(1.+torch.exp(-1*steepness*(x-0.5)))

    def get_remaining(self, steepness=20.):
        """return the fraction of active zeta_t (i.e > 0.5)"""
        def n_remaining(m):
            return (m.pruned_zeta if m.is_pruned else self.smoothRound(m.get_zeta_t(), steepness)).sum()  
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
        plt.hist(zetas)
        plt.show()
        return exactly_zeros, exactly_ones
    
    def get_crispnessLoss(self, device):
        """loss reponsible for making zeta_t 1 or 0"""
        loss = torch.FloatTensor([]).to(device)
        for l_block in self.modules():
            if isinstance(l_block, PrunableBatchNorm2d):
                loss = torch.cat([loss, torch.pow(l_block.get_zeta_t()-l_block.get_zeta_i(), 2)])
        return torch.mean(loss)

    def prune(self, Vc, finetuning=False, threshold=None):
        """prunes the network to make zeta_t exactly 1 and 0"""
        if threshold==None:
            self.prune_threshold = self.calculate_prune_threshold(Vc)
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

    def unprune():
        for l_block in self.modules():
            if isinstance(l_block, PrunableBatchNorm2d):
                l_block.unprune()
    
    def prepare_for_finetuning(self, beta, gamma, device, Vc):
        """freezes zeta"""
        self.device = device
        self.set_beta_gamma(beta, gamma, device)

        threshold = self.prune(Vc, finetuning=True)
        while self.get_remaining()<Vc:
            threshold-=0.0001
            self.prune(Vc, finetuning=True, threshold=threshold)

        return threshold            
          
    def set_beta_gamma(self, beta, gamma, device):
        for l_block in self.modules():
            if isinstance(l_block, PrunableBatchNorm2d):
                l_block.set_beta_gamma(beta, gamma, device)
    
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