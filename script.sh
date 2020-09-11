#!/bin/bash

python pretraining.py $0 $1 --epohcs 300
python pruning.py $0 $1 --Vc 0.0625
python finetuning --name $1\_$0\_0.0625 --epochs 300

python pruning.py $0 $1 --Vc 0.125
python finetuning --name $1\_$0\_0.125 --epochs 300

python pruning.py $0 $1 --Vc 0.25
python finetuning --name $1\_$0\_0.25 --epochs 300

python pruning.py $0 $1 --Vc 0.5
python finetuning --name $1\_$0\_0.5 --epochs 300