import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import os
import random
import pandas as pd
# from resnet import PrunableBatchNorm2d
def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
    img = img_denorm(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
    
def img_denorm(img):
    #for ImageNet the mean and std are:
    mean = np.asarray([ 0.485, 0.456, 0.406 ])
    std = np.asarray([ 0.229, 0.224, 0.225 ])

    denormalize = transforms.Normalize((-1 * mean / std), (1.0 / std))

    res = img.squeeze(0)
    res = denormalize(res)
    res = torch.clamp(res, 0, 1)    
    return(res)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = 0.05 * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
class MyReLU(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return torch.max(0.25*input,torch.min(4*input-1.5,0.25*input+0.75))

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()*4.
        grad_input[input < 0.4] = 0.25
        grad_input[input > 0.6] = 0.25
        return grad_input
        
def plot_learning_curves(logger_name):
    train_loss = []
    val_loss = []
    val_acc = []
    df = pd.read_csv('logs/'+logger_name)
#     print(df.head())
#     return
    train_loss = df.iloc[1:,1]
    val_loss = df.iloc[1:,2]
    val_acc = df.iloc[1:,3]*100
    
    plt.style.use('seaborn')
    plt.plot(np.arange(len(train_loss)), train_loss, label = 'Training error')
    plt.plot(np.arange(len(train_loss)), val_loss, label = 'Validation error')
    plt.ylabel('Loss', fontsize = 14)
    plt.xlabel('Epochs', fontsize = 14)
    plt.title('Loss Curve', fontsize = 18, y = 1.03)
    plt.legend()
    plt.ylim(0,4)
    plt.show()
    print()
    
    plt.style.use('seaborn')
    plt.plot(np.arange(len(train_loss)), val_acc, label = 'Validation Accuracy')
    plt.ylabel('Accuracy', fontsize = 14)
    plt.xlabel('Epochs', fontsize = 14)
    plt.title('Accuracy curve', fontsize = 18, y = 1.03)
    plt.legend()
    plt.ylim(0,100)
    plt.show()
    print()

def visualize_model_architecture(model):
    pruned_model = [3,]
    full_model = [3,]
    for l_block in model.modules():
        if hasattr(l_block, 'zeta'):
            zeta = l_block.get_zeta_t().cpu().detach().numpy().tolist()
            full_model.append(len(zeta))
            pruned_model.append(np.sum(zeta))
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    full_model = np.array(full_model)
    pruned_model = np.array(pruned_model)
    ax.bar(np.arange(len(full_model)), full_model, width = 0.5, color = 'b')
    ax.bar(np.arange(len(pruned_model)), pruned_model, width = 0.5, color = 'r')
    print(full_model)
    print(pruned_model)
    plt.show()
    return [full_model, pruned_model]

        