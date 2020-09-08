import argparse
import glob
import os
import sys


import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.hub import load_state_dict_from_url
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from tqdm import tqdm as tqdm_notebook
import random 
import os

from utils import *
from models import get_model

seed_everything(43)

ap = argparse.ArgumentParser(description='pruning with heaviside continuous approximations and logistic curves')
ap.add_argument('dataset', choices=['c10', 'c100','tin'], type=str, help='Dataset choice')
ap.add_argument('model', choices=['wrn','r50'], type=str, help='Model choice')
ap.add_argument('--Vc', default=0.25, type=float, help='Budget Constraint')
ap.add_argument('--batch_size', default=32, type=int, help='Batch Size')
ap.add_argument('--epochs', default=20, type=int, help='Epochs')
ap.add_argument('--workers', default=0, type=int, help='Number of CPU workers')
ap.add_argument('--valid_size', '-v', type=float, default=0.1, help='valid_size')
ap.add_argument('--lr', default=0.001, type=float, help='Learning rate')
ap.add_argument('--test_only','-t', default=False, type=bool, help='Testing')

ap.add_argument('--decay', default=0.001, type=float, help='Weight decay')
ap.add_argument('--w1', default=30., type=float, help='weightage to budget loss')
ap.add_argument('--w2', default=10., type=float, help='weightage to crispness loss')
ap.add_argument('--b_inc', default=5., type=float, help='beta increment')
ap.add_argument('--g_inc', default=2., type=float, help='gamma increment')

ap.add_argument('--cuda_id', '-id', type=str, default='0', help='gpu number')
args = ap.parse_args()

valid_size=args.valid_size
BATCH_SIZE = args.batch_size
Vc = torch.FloatTensor([args.Vc])
MODEL = args.model

############################### preparing dataset ################################

data_object = data.DataManager(args)
trainloader, valloader, testloader = data_object.prepare_data()
dataloaders = {
        'train': trainloader, 'val': valloader, "test": testloader
}

############################### preparing model ###################################

model = get_model(args.model,'prune',data_object.num_classes)
pruned_model = get_model(args.model,'prune',data_object.num_classes)
state = torch.load(f"models/{args.model+args.dataset}_pretrained.pth")
model.load_state_dict(state['state_dict'],strict=False)

############################### preparing for pruning ###################################

if os.path.exists('logs') == False:
    os.mkdir("logs")

if os.path.exists('checkpoints') == False:
    os.mkdir("checkpoints")


weightage1 = args.w1 #weightage given to budget loss
weightage2 = args.w2 #weightage given to crispness loss
steepness = 10. # steepness of gate_approximator

CE = nn.CrossEntropyLoss()
def loss1(model, y_pred, y_true, epoch):
    ce_loss = CE(y_pred, y_true).to(device)
    budget_loss = ((model.get_remaining(steepness).to(device)-Vc.to(device))**2).to(device)
    crispness_loss =  model.get_zt_diff_zi_sq_loss(device).to(device)
    return  budget_loss*weightage1 + crispness_loss*weightage2 + ce_loss

criterion = loss1
param_optimizer = list(model.named_parameters())
no_decay = ["zeta"]
optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.decay,'lr':args.lr},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,'lr':args.lr*0.1},
    ]
optimizer = optim.AdamW(optimizer_parameters)

device = torch.device(f"cuda:{str(args.cuda_id)}")
model.to(device)
pruned_model.to(device)
Vc.to(device)



def train(model, loss_fn, optimizer,epoch):
    global steepness
    model.train()
    counter = 0
    tk1 = tqdm_notebook(dataloaders['train'], total=len(dataloaders['train']))
    running_loss = 0
    for x_var, y_var in tk1:
        counter +=1
        x_var = x_var.to(device=device)
        y_var = y_var.to(device=device)
        scores = model(x_var)
        loss = loss_fn(model,scores, y_var,epoch)
        optimizer.zero_grad()
        loss.backward()
        running_loss+=loss.item()
        tk1.set_postfix(loss=(running_loss /counter))
        optimizer.step()
        steepness += (50./(5*len(tk1)))
    return (running_loss/counter)        


def test(model, loss_fn, optimizer, phase,epoch):
    model.eval()
    counter = 0
    tk1 = tqdm_notebook(dataloaders[phase], total=len(dataloaders[phase]))
    running_loss = 0
    running_acc = 0
    total = 0
    with torch.no_grad():
        for x_var, y_var in tk1:
            counter +=1
            x_var = x_var.to(device=device)
            y_var = y_var.to(device=device)
            scores = model(x_var)
            loss = loss_fn(model,scores, y_var,epoch)
            _, scores = torch.max(scores.data, 1)
            y_var = y_var.cpu().detach().numpy()
            scores = scores.cpu().detach().numpy()
            
            correct = (scores == y_var).sum().item()
            running_loss+=loss.item()
            running_acc+=correct
            total += scores.shape[0]
            tk1.set_postfix(loss=(running_loss /counter), acc=(running_acc/total))
    return (running_acc/total)

best_acc=0
beta, gamma = 1., 2.
model.set_beta_gamma(beta, gamma, device)

rem_bef_pru = []
rem_after_pru = []
val_acc = []
pru_acc = []
pru_thresh = []
ex_zeros = []
ex_ones = []
problems = []
name = f'{args.model+args.dataset}_{args.w1}_{args.w2}_{args.w3}_{args.b_inc}_{args.g_inc}_{args.decay}_{str(Vc.item())}_pruned_{args.counter}'
if(args.test == False):
    for epoch in range(args.epochs):
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        train(model, criterion, optimizer,epoch)
        print('----------validation before pruning---------')
        acc1 = test(model, criterion, optimizer, "val", epoch)
        rem = model.get_remaining(steepness).item()
        print("Rem:", rem)
        exactly_zeros, exactly_ones = model.plot_zt()
        rem_bef_pru.append(rem)
        val_acc.append(acc1)
        ex_zeros.append(exactly_zeros)
        ex_ones.append(exactly_ones)
        
        
        print('----------validation after pruning---------')
        pruned_model.load_state_dict(model.state_dict())
        thresh, problem = pruned_model.prune_net(args.Vc)
        
        acc = test(pruned_model, criterion, optimizer, "test",epoch)
        rem = pruned_model.get_remaining().item()
        print("Rem:", rem)
        print("FATAL: ", problem)
        pru_acc.append(acc)
        pru_thresh.append(thresh)
        rem_after_pru.append(rem)
        problems.append(problem)
        
        if (epoch+1)%1==0:
            beta=min(6., beta+(0.1/args.b_inc))
            gamma=min(256, gamma*(2**(1./args.g_inc)))
            model.set_beta_gamma(beta, gamma, device)
            print("Changed beta to", beta, "changed gamma to", gamma)
            
        
        if (acc>best_acc and rem<=(Vc.item()+0.01)):
            print("**Saving model**")
            best_acc=acc
            torch.save({
                "epoch" : epoch+1,
                "beta" : beta,
                "gamma" : gamma,
                "rem" : rem,
                "prune_thresh":thresh,
                "state_dict" : model.state_dict(),
                "acc" : acc,
            }, f"checkpoints/{name}.pth")


        df_data=np.array([rem_bef_pru,rem_after_pru,val_acc,pru_acc,pru_thresh,ex_zeros,ex_ones,problems]).T
        df = pd.DataFrame(df_data,columns = ['rem_bef_pru','rem_after_pru','val_acc','pru_acc','pru_thresh','ex_zeros','ex_ones','problems'])
        df.to_csv(f"logs/{name}.csv")