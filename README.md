
# Pruning-Deep-Networks-with-heaviside-continuous-approximations-and-logistic-curves

This repo is the official code for CHIPNET: Budget-Aware Pruning with Heaviside Continuous Approximations

## Prerequisites

Install prerequisites with:  
```
pip install -r requirements.txt
```

# Usage (summary)

The main script offers many options; here are the most important ones:

### Pretraining

```
usage: python pretraining.py [c10, c100, tin] [wrn, r50, r101]
```
```
usage: python pretraining.py [c10, c100] [r164] --epochs 160 --decay 0.001 --batch_size 64 --lr 0.1 --scheduler_type 2
```
### Pruning
```
usage: python pruning.py [c10, c100, tin] [wrn] --Vc {float value between 0-1} --budget_type {channel_ratio, volume_ratio, parameter_ratio, flops_ratio} --epochs 20
```

### Finetuning
```
usage: python finetuning.py [c10, c100, tin] [wrn, r50, r101] --Vc {float value between 0-1} --budget_type {channel_ratio, volume_ratio, parameter_ratio, flops_ratio} --name {model name}_{dataset}_{budget}_{budget_type}
```
```
usage: python finetuning.py [c10, c100, tin] [r164] --Vc {float value between 0-1} --budget_type {channel_ratio, volume_ratio, parameter_ratio, flops_ratio} --name {model name}_{dataset}_{budget}_{budget_type} --epochs 160 --decay 0.001 --batch_size 64 --lr 0.1 --scheduler_type 2
```
### Mask Transfer
```
usage: python finetuning.py [c10, c100, tin] [wrn, r50, r101] --Vc {float value between 0-1} --budget_type {channel_ratio, volume_ratio, parameter_ratio, flops_ratio} --host_name {model name}_{dataset}_{budget}_{budget_type}
```
```
usage: python finetuning.py [c10, c100, tin] [r164] --Vc {float value between 0-1} --budget_type {channel_ratio, volume_ratio, parameter_ratio, flops_ratio} --host_name {model name}_{dataset}_{budget}_{budget_type} --epochs 160 --decay 0.001 --batch_size 64 --lr 0.1 --scheduler_type 2
```


***
* Parameter and FLOPs budget is supported only with WRN for now.
