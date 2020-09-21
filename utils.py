import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import random
import pandas as pd

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_mask_dict(own_state, state_dict):
    for name, param in state_dict.items():
        if name not in own_state:
             continue
        if 'zeta' not in name and 'beta' not in name and 'gamma' not in name:
            continue
        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)
    return own_state
    
def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    if args.scheduler_type==1:
        lr = args.lr * (0.5 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        if epoch in [args.epochs*0.5, args.epochs*0.75]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        
def plot_learning_curves(logger_name):
    train_loss = []
    val_loss = []
    val_acc = []
    df = pd.read_csv('logs/'+logger_name)

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

def visualize_model_architecture(model, budget, budget_type):
    pruned_model = [3,]
    full_model = [3,]
    device = torch.device('cpu')
    model.to(device)
    model(torch.rand(1,3,32,32))
    model.prepare_for_finetuning(device=device,budget=budget,budget_type=budget_type)
    for l_block in model.prunable_modules:
        gates = l_block.pruned_zeta.cpu().detach().numpy().tolist()
        full_model.append(len(gates))
        pruned_model.append(np.sum(gates))
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    full_model = np.array(full_model)
    pruned_model = np.array(pruned_model)
    ax.bar(np.arange(len(full_model)), full_model, width = 0.5, color = 'b')
    ax.bar(np.arange(len(pruned_model)), pruned_model, width = 0.5, color = 'r')
    print(full_model)
    print(pruned_model)
    plt.show()
    active_params, total_params = model.get_params_count()
    
    print(f'Total parameter count: {total_params}')
    print(f'Remaining parameter count: {active_params}')
    print(f'Remaining Parameter Fraction: {active_params/total_params}')
    return [full_model, pruned_model]
        