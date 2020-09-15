import argparse
import os

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm as tqdm_notebook
from datasets import DataManager
from utils import *
from models import get_model

seed_everything(43)

ap = argparse.ArgumentParser(description='finetuning')
ap.add_argument('dataset', choices=['c10', 'c100', 'tin'], type=str, help='Dataset choice')
ap.add_argument('model', choices=['wrn', 'r32', 'r50', 'r101', 'r152', 'r164', 'vgg11', 'vgg13', 'vgg16', 'vgg19'], type=str, help='Model choice')
ap.add_argument('--budget_type', choices=['channel_ratio', 'volume_ratio'], type=str, help='Budget Type')
ap.add_argument('--Vc', default=0.5, type=float, help='Budget Constraint')
ap.add_argument('--batch_size', default=128, type=int, help='Batch Size')
ap.add_argument('--epochs', default=200, type=int, help='Epochs')
ap.add_argument('--name', type=str, help='name of model')

ap.add_argument('--valid_size', '-v', type=float, default=0.1, help='valid_size')
ap.add_argument('--lr', default=0.05, type=float, help='Learning rate')
ap.add_argument('--test_only', '-t', type=bool, default=False, help='test the best model')
ap.add_argument('--workers', default=0, type=int, help='number of workers')
ap.add_argument('--cuda_id', '-id', type=str, default='0', help='gpu number')
args = ap.parse_args()

valid_size=args.valid_size
Vc = torch.FloatTensor([args.Vc])
model_path = f"checkpoints/{args.name}_pruned.pth"

############################### preparing dataset ################################

data_object = DataManager(args)
trainloader, valloader, testloader = data_object.prepare_data()
dataloaders = {
        'train': trainloader, 'val': valloader, "test": testloader
}

############################### preparing model ###################################

model = get_model(args.model, 'prune', data_object.num_classes)
        
state = torch.load(model_path)
model.load_state_dict(state['state_dict'], strict=False)

CE = nn.CrossEntropyLoss()
def criterion(model, y_pred, y_true):
    ce_loss = CE(y_pred, y_true)
    return ce_loss

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)
device = torch.device(f"cuda:{str(args.cuda_id)}")
model.to(device)
Vc.to(device)

def train(model, loss_fn, optimizer):
    model.train()
    counter = 0
    tk1 = tqdm_notebook(dataloaders['train'], total=len(dataloaders['train']))
    running_loss = 0.
    for x_var, y_var in tk1:
        counter +=1
        x_var = x_var.to(device=device)
        y_var = y_var.to(device=device)
        scores = model(x_var)
        loss = loss_fn(model, scores, y_var)
        running_loss+=loss.item()
        tk1.set_postfix(loss=running_loss/counter)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return running_loss/counter        

def test(model, loss_fn, optimizer, phase):
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
            loss = loss_fn(model, scores, y_var)
            _, scores = torch.max(scores.data, 1)
            y_var = y_var.cpu().detach().numpy()
            scores = scores.cpu().detach().numpy()

            correct = (scores == y_var).sum().item()
            running_loss+=loss.item()
            running_acc+=correct
            total+=scores.shape[0]
            tk1.set_postfix(loss=running_loss/counter, acc=running_acc/total)
    return running_acc/total, running_loss/counter

############################## training starts here #############################

model.prepare_for_finetuning(device, Vc.item(), budget_type=args.budget_type) # sets beta and gamma and unfreezes network except zetas

best_accuracy=0
num_epochs = args.epochs
train_losses = []
valid_losses = []
valid_accuracy = []
if args.test_only == False:
    for epoch in range(num_epochs):
        adjust_learning_rate(optimizer, epoch)
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        train_loss = train(model, criterion, optimizer)
        accuracy, valid_loss = test(model, criterion, optimizer, "val")
        remaining = model.get_remaining(20.,args.budget_type).item()
        
        if accuracy>best_accuracy:
            print("**Saving model**")
            best_accuracy=accuracy
            torch.save({
                "epoch": epoch + 1,
                "state_dict" : model.state_dict(),
                "acc" : best_accuracy,
                "rem" : remaining,
            }, f"checkpoints/{args.name}_{args.dataset}_finetuned.pth")
            
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_accuracy.append(accuracy)
        df_data=np.array([train_losses, valid_losses, valid_accuracy]).T
        df = pd.DataFrame(df_data,columns = ['train_losses','valid_losses','valid_accuracy'])
        df.to_csv(f"logs/{args.name}_{args.dataset}_finetuned.csv")

state = torch.load(f"checkpoints/{args.name}_{args.dataset}_finetuned.pth")
model.load_state_dict(state['state_dict'],strict=True)
acc, v_loss = test(model, criterion, optimizer, "test")
print(f"Test Accuracy: {acc} | Valid Accuracy: {state['acc']}")