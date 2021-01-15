# ChipNet

This is the official repository to the ICLR 2021 paper "[ChipNet: Budget-Aware Pruning with Heaviside Continuous Approximations](
https://openreview.net/pdf?id=xCxXwTzx4L1)" by Rishabh Tiwari, Udbhav Bamba, Arnav Chavan, Deepak Gupta.

## Getting Started

You will need [Python 3.7](https://www.python.org/downloads) and the packages specified in _requirements.txt_.
We recommend setting up a [virtual environment with pip](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
and installing the packages there.

Install packages with:

```
$ pip install -r requirements.txt
```

## Configure and Run

All configurations concerning data, model, training, etc. can be called using commandline arguments.

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
* Parameter and FLOPs budget is supported only with models using ResNetCifar module for now.

## Citation
Please cite our paper in your publications if it helps your research. Even if it does not,and you want to make us happy, do cite it :)

    @inproceedings{
    tiwari2021chipnet,
    title={ChipNet: Budget-Aware Pruning with Heaviside Continuous Approximations},
    author={Rishabh Tiwari and Udbhav Bamba and Arnav Chavan and Deepak Gupta},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=xCxXwTzx4L1}
    }
    


## License

This project is licensed under the MIT License.
