#!/bin/bash
echo 
# python pretraining.py $1 $2 --epochs 160 --batch_size 64
# python pruning.py $1 $2 --Vc 0.0625 --budget_type 'parameter_ratio'
python finetuning.py $1 $2 --name $2\_$1\_0.0625\_parameter\_ratio --epochs 300 --Vc 0.0625 --budget_type parameter_ratio
# python pruning.py $1 $2 --Vc 0.4
# python pruning.py $1 $2 --Vc 0.2
# python pruning.py $1 $2 --Vc 0.1 --w1 45.

# python finetuning.py $1 $2 --name $2\_$1\_0.6\_channel\_ratio --epochs 160 --Vc 0.6 --batch_size 64
# python finetuning.py $1 $2 --name $2\_$1\_0.4\_channel\_ratio --epochs 160 --Vc 0.4 --batch_size 64
# python finetuning.py $1 $2 --name $2\_$1\_0.2\_channel\_ratio --epochs 160 --Vc 0.2 --batch_size 64
# python finetuning.py $1 $2 --name $2\_$1\_0.1\_channel\_ratio --epochs 160 --Vc 0.1 --batch_size 64