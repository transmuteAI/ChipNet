#!/bin/bash
echo enter model name
read model
echo enter dataset
read dataset
echo enter budget
read budget

python pretraining.py $dataset $model --epochs 300
python pruning.py $dataset $model --budget $budget
python finetuning --name $model\_$dataset\_$budget --epochs 300 
