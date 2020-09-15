#!/bin/bash
echo 
# python pretraining.py $1 $2 --epochs 300
python pruning.py $1 $2 --Vc 0.0625 --budget_type volume_ratio
python finetuning.py $1 $2 --name $2\_$1\_0.0625\_volume_ratio --epochs 300 --Vc 0.0625 --budget_type volume_ratio

python pruning.py $1 $2 --Vc 0.125 --budget_type volume_ratio
python finetuning.py $1 $2 --name $2\_$1\_0.125\_volume_ratio --epochs 300 --Vc 0.125 --budget_type volume_ratio

python pruning.py $1 $2 --Vc 0.25 --budget_type volume_ratio
python finetuning.py $1 $2 --name $2\_$1\_0.25\_volume_ratio --epochs 300 --Vc 0.25 --budget_type volume_ratio

python pruning.py $1 $2 --Vc 0.5 --budget_type volume_ratio
python finetuning.py $1 $2 --name $2\_$1\_0.5\_volume_ratio --epochs 300 --Vc 0.5 --budget_type volume_ratio