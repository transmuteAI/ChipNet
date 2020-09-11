#!/bin/bash
echo enter model name
read model
echo enter dataset
read dataset
echo enter budget
read budget

python pruning.py $dataset $model --Vc $budget
python finetuning --name $model\_$dataset\_$budget --epochs 300 
