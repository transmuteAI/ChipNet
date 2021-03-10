import argparse
import os

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm as tqdm_notebook

from utils import *
from models import get_model
from datasets import DataManager

seed_everything(43)

ap = argparse.ArgumentParser(description='pruning with heaviside continuous approximations and logistic curves')
ap.add_argument('dataset', choices=['c10', 'c100', 'tin','svhn'], type=str, help='Dataset choice')
ap.add_argument('model', type=str, help='Model choice')
ap.add_argument('--budget_type', choices=['channel_ratio', 'volume_ratio','parameter_ratio','flops_ratio'], default='channel_ratio', type=str, help='Budget Type')
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

valid_size = args.valid_size
BATCH_SIZE = args.batch_size
Vc = torch.FloatTensor([args.Vc])

############################### preparing dataset ################################

data_object = DataManager(args)
trainloader, valloader, testloader = data_object.prepare_data()
dataloaders = {
        'train': trainloader, 'val': valloader, "test": testloader
}

############################### preparing model ###################################

model = get_model(args.model, 'prune', data_object.num_classes, data_object.insize)
state = torch.load(f"checkpoints/{args.model}_{args.dataset}_pretrained.pth")
model.load_state_dict(state['state_dict'], strict=False)

############################### preparing for pruning ###################################

if os.path.exists('logs') == False:
    os.mkdir("logs")

if os.path.exists('checkpoints') == False:
    os.mkdir("checkpoints")


weightage1 = args.w1 #weightage given to budget loss
weightage2 = args.w2 #weightage given to crispness loss

CE = nn.CrossEntropyLoss()
def criterion(model, y_pred, y_true):
    ce_loss = CE(y_pred, y_true)
    budget_loss = ((model.get_remaining(args.budget_type).to(device)-Vc.to(device))**2).to(device)
    crispness_loss =  model.get_crispnessLoss(device)
    return budget_loss*weightage1 + crispness_loss*weightage2 + ce_loss

param_optimizer = list(model.named_parameters())
no_decay = ["zeta"]
optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.decay,'lr':args.lr},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,'lr':args.lr},
    ]
optimizer = optim.AdamW(optimizer_parameters)

device = torch.device(f"cuda:{str(args.cuda_id)}")
model.to(device)
Vc.to(device)

def train(model, loss_fn, optimizer, epoch):
    model.train()
    counter = 0
    tk1 = tqdm_notebook(dataloaders['train'], total=len(dataloaders['train']))
    running_loss = 0
    for x_var, y_var in tk1:
        counter +=1
        x_var = x_var.to(device=device)
        y_var = y_var.to(device=device)
        scores = model(x_var)
        loss = loss_fn(model,scores, y_var)
        optimizer.zero_grad()
        loss.backward()
        running_loss+=loss.item()
        tk1.set_postfix(loss=running_loss/counter)
        optimizer.step()
    return running_loss/counter

def test(model, loss_fn, optimizer, phase, epoch):
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
            loss = loss_fn(model,scores, y_var)
            _, scores = torch.max(scores.data, 1)
            y_var = y_var.cpu().detach().numpy()
            scores = scores.cpu().detach().numpy()
            
            correct = (scores == y_var).sum().item()
            running_loss+=loss.item()
            running_acc+=correct
            total+=scores.shape[0]
            tk1.set_postfix(loss=(running_loss /counter), acc=(running_acc/total))
    return running_acc/total

best_acc = 0
beta, gamma = 1., 2.
model.set_beta_gamma(beta, gamma)

remaining_after_pruning = []
valid_accuracy = []
problems = []
name = f'{args.model}_{args.dataset}_{str(np.round(Vc.item(),decimals=6))}_{args.budget_type}_pruned'
if args.test_only == False:
    for epoch in range(args.epochs):
        print(f'Starting epoch {epoch + 1} / {args.epochs}')
        model.unprune()
        train(model, criterion, optimizer, epoch)
        print(f'[{epoch + 1} / {args.epochs}] Validation')
        acc = test(model, criterion, optimizer, "val", epoch)
        problem = model.prune(args.Vc, args.budget_type)
        remaining = model.get_remaining(args.budget_type).item()

        remaining_after_pruning.append(remaining)
        valid_accuracy.append(acc)
        problems.append(problem)

        beta=min(6., beta+(0.1/args.b_inc))
        gamma=min(256, gamma*(2**(1./args.g_inc)))
        model.set_beta_gamma(beta, gamma)
        print("Changed beta to", beta, "changed gamma to", gamma)

        if acc>best_acc:
            print("**Saving checkpoint**")
            best_acc=acc
            torch.save({
                "epoch" : epoch+1,
                "beta" : beta,
                "gamma" : gamma,
                "state_dict" : model.state_dict(),
                "accuracy" : acc,
            }, f"checkpoints/{name}.pth")

        df_data=np.array([remaining_after_pruning, valid_accuracy, problems]).T
        df = pd.DataFrame(df_data,columns = ['Remaining after pruning', 'Valid accuracy', 'problems'])
        df.to_csv(f"logs/{name}.csv")