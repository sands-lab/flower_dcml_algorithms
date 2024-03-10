#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=15:00:00
#SBATCH --cpus-per-task=42
#SBATCH --gpus=2
#SBATCH --output=logs/txt/grid_search_fd.txt

source ~/.venv/flower/bin/activate
source ./grid_search_scripts/common.sh

export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1

## grid search for FedAvg
batch_size=8
lr=0.05


for local_epochs in 1 2 4; do
    for temperature in 1.0 2.0; do
        for kd_strength in 0.1 0.2 0.5; do
            local_to_global_epochs_mapping $local_epochs

            echo "$lr $local_epochs $batch_size"
            python fl.py fl_algorithm=fd \
                local_train.lr=$lr \
                local_train.local_epochs=$local_epochs \
                local_train.batch_size=$batch_size \
                global_train.epochs=$global_epochs \
                fl_algorithm.client.kd_weight=$kd_strength \
                fl_algorithm.client.temperature=$temperature \
                logging.name_keys=[local_train.local_epochs,fl_algorithm.client.kd_weight,fl_algorithm.client.temperature]
            echo "========================================"
            echo "========================================"
            echo "========================================"
        done
    done
done
