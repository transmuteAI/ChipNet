
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
usage: python pretraining.py [dataset] [model name] --epochs [number of epochs] --decay [weight decay] --batch_size [batch size] --lr [learning rate] --scheduler_type {1, 2}
```

### Pruning
```
usage: python pruning.py [dataset] [model name] --Vc {float value between 0-1} --budget_type {channel_ratio, volume_ratio, parameter_ratio, flops_ratio} --epochs 20
```

### Finetuning
```
usage: python finetuning.py [dataset] [model name] --Vc {float value between 0-1} --budget_type {channel_ratio, volume_ratio, parameter_ratio, flops_ratio} --name {model name}_{dataset}_{budget}_{budget_type} --epochs [number of epochs] --decay [weight decay] --batch_size [batch size] --lr [learning rate] --scheduler_type {1, 2}
```

### Mask Transfer
```
usage: python finetuning.py [dataset] [model name] --Vc {float value between 0-1} --budget_type {channel_ratio, volume_ratio, parameter_ratio, flops_ratio} --host_name {model name}_{dataset}_{budget}_{budget_type} --epochs [number of epochs] --decay [weight decay] --batch_size [batch size] --lr [learning rate] --scheduler_type {1, 2}
```


***
* Parameter and FLOPs budget is supported only with models using ResNetCifar module for now.
