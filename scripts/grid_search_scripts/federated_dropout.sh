#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=42
#SBATCH --gpus=2
#SBATCH --output=logs/txt/grid_search_federated_dropout.txt

source ~/.venv/flower/bin/activate
source ./grid_search_scripts/common.sh

export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1

## grid search for FedAvg

for local_epochs in 1 2 4; do
    for lr in 0.02 0.05 0.1; do
        for batch_size in 8 16; do

            local_to_global_epochs_mapping $local_epochs

            echo "$lr $local_epochs $batch_size"
            python -u fl.py fl_algorithm=federated_dropout \
                local_train.lr=$lr \
                local_train.local_epochs=$local_epochs \
                global_train.epochs=$global_epochs \
                local_train.batch_size=$batch_size \
                logging.name_keys=[local_train.lr,local_train.local_epochs,local_train.batch_size]
            echo "========================================"
            echo "========================================"
            echo "========================================"
        done
    done
done
