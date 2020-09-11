#!/bin/bash
echo 
python pretraining.py $1 $2 --epochs 300
python pruning.py $1 $2 --Vc 0.0625
python finetuning.py $1 $2 --name $2\_$1\_0.0625 --epochs 300

python pruning.py $1 $2 --Vc 0.125
python finetuning.py $1 $2 --name $2\_$1\_0.125 --epochs 300

python pruning.py $1 $2 --Vc 0.25
python finetuning.py $1 $2 --name $2\_$1\_0.25 --epochs 300

python pruning.py $1 $2 --Vc 0.5
python finetuning.py $1 $2 --name $2\_$1\_0.5 --epochs 300