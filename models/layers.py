import torch
import torch.nn as nn


class PrunableBatchNorm2d(torch.nn.BatchNorm2d):
    def __init__(self, num_features, conv_module=None):
        super(PrunableBatchNorm2d, self).__init__(num_features=num_features)
        self.is_imp = False
        self.is_pruned = False
        self.num_gates = num_features
        self.zeta = nn.Parameter(torch.rand(num_features) * 0.01)
        self.pruned_zeta = torch.ones_like(self.zeta)
        beta=1.
        gamma=2.
        for n, x in zip(('beta', 'gamma'), (torch.tensor([x], requires_grad=False) for x in (beta, gamma))):
            self.register_buffer(n, x)  # self.beta will be created (same for gamma, zeta)        
    
    def forward(self, input):
        out = super(PrunableBatchNorm2d, self).forward(input)
        z = self.pruned_zeta if self.is_pruned else self.get_zeta_t()
        out *= z[None, :, None, None] # broadcast the mask to all samples in the batch, and all locations
        return out
    
    def get_zeta_i(self):
        return self.__generalized_logistic(self.zeta)
    
    def get_zeta_t(self):
        zeta_i = self.get_zeta_i()
        return self.__continous_heavy_side(zeta_i)

    def set_beta_gamma(self, beta, gamma):
        self.beta.data.copy_(torch.Tensor([beta]))
        self.gamma.data.copy_(torch.Tensor([gamma]))
      
    def __generalized_logistic(self, x):
        return 1./(1.+torch.exp(-self.beta*x))
    
    def __continous_heavy_side(self, x):
        return 1-torch.exp(-self.gamma*x)+x*torch.exp(-self.gamma)
    
    def prune(self, threshold):
        self.is_pruned = True
        self.pruned_zeta = (self.get_zeta_t()>threshold).long()
        self.zeta.requires_grad = False

    def unprune(self):
        self.is_pruned = False
        self.zeta.requires_grad = True

    @staticmethod
    def from_batchnorm(bn_module, conv_module):
        new_bn = PrunableBatchNorm2d(bn_module.num_features, conv_module)
        return new_bn, conv_module


class ModuleInjection:
    pruning_method = 'full'
    prunable_modules = []

    @staticmethod
    def make_prunable(conv_module: nn.Conv2d, bn_module: nn.BatchNorm2d):
        """Make a (conv, bn) sequence prunable.
        :param conv_module: A Conv2d module
        :param bn_module: The BatchNorm2d module following the Conv2d above
        :param prune_before_bn: Whether the pruning gates will be applied before or after the Batch Norm
        :return: a pair (conv, bn) that can be trained to
        """
        if ModuleInjection.pruning_method == 'full':
            return conv_module, bn_module
        new_bn, conv_module = PrunableBatchNorm2d.from_batchnorm(bn_module, conv_module=conv_module)
        ModuleInjection.prunable_modules.append(new_bn)
        return conv_module, new_bn