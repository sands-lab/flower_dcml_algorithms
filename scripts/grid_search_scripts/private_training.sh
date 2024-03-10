#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=15:00:00
#SBATCH --cpus-per-task=42
#SBATCH --gpus=2
#SBATCH --output=logs/txt/grid_search_private_training.txt

source ~/.venv/flower/bin/activate

export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1

local_epochs=1

## grid search for FedAvg
for lr in 0.01 0.05 0.1; do
    for batch_size in 8 16; do
        for weight_decay in 0.001 0.0005 0.0001; do

            echo "$lr $local_epochs $batch_size"
            python fl.py fl_algorithm=private_training \
                local_train.lr=$lr \
                local_train.local_epochs=$local_epochs \
                local_train.batch_size=$batch_size \
                fl_algorithm.client.weight_decay=$weight_decay \
                logging.name_keys=[local_train.lr,local_train.batch_size,fl_algorithm.client.weight_decay]
            echo "========================================"
            echo "========================================"
        done
    done
done
