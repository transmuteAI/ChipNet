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
        if conv_module is not None:
            def fo_hook(module, in_tensor, out_tensor):
                module.num_input_active_channels = (in_tensor[0].sum((0,2,3))>0).sum().item()
                module.output_area = out_tensor.size(2) * out_tensor.size(3)
            conv_module.register_forward_hook(fo_hook)
        self._conv_module = conv_module
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
        self.pruned_zeta = (self.get_zeta_t()>threshold).float()
        # if self.is_imp and self.pruned_zeta.sum()==0:
        #     self.pruned_zeta[torch.argmax(self.get_zeta_t()).item()] = 1.
        self.zeta.requires_grad = False

    def unprune(self):
        self.is_pruned = False
        self.zeta.requires_grad = True

    def get_params_count(self):
        total_conv_params = self._conv_module.in_channels*self.pruned_zeta.shape[0]*self._conv_module.kernel_size[0]*self._conv_module.kernel_size[1]
        bn_params = self.num_gates*2
        active_bn_params = self.pruned_zeta.sum().item()*2
        active_conv_params = self._conv_module.num_input_active_channels*self.pruned_zeta.sum().item()*self._conv_module.kernel_size[0]*self._conv_module.kernel_size[1]
        return active_conv_params+active_bn_params, total_conv_params+bn_params

    def get_volume(self):
        total_volume = self._conv_module.output_area*self.num_gates
        active_volume = self._conv_module.output_area*self.pruned_zeta.sum().item()
        return active_volume, total_volume
    
    def get_flops(self):
        k_area = m.kernel_size[0]*m.kernel_size[1]
        total_flops = self._conv_module.output_area*self.num_gates*self._conv_module.in_channels*k_area
        active_flops = self._conv_module.output_area*self.pruned_zeta.sum().item()*self._conv_module.num_input_active_channels*k_area
        return active_flops, total_flops

    @staticmethod
    def from_batchnorm(bn_module, conv_module):
        new_bn = PrunableBatchNorm2d(bn_module.num_features, conv_module)
        return new_bn, conv_module


class ModuleInjection:
    pruning_method = 'full'
    prunable_modules = []

    @staticmethod
    def make_prunable(conv_module, bn_module):
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