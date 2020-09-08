import torch
import torch.nn as nn
import numpy as np

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