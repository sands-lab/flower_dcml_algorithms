#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=42
#SBATCH --gpus=2
#SBATCH --output=logs/txt/grid_search_lg_fedavg_large_encoder.txt

source ~/.venv/flower/bin/activate
source ./grid_search_scripts/common.sh

export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1

## grid search for FedAvg
lr=0.05
batch_size=8

for lte in 1 2 4; do
    local_to_global_epochs_mapping $lte

    echo "$lr $local_epochs $batch_size"
    python fl.py fl_algorithm=lg_fedavg \
        local_train.lr=$lr \
        local_train.local_epochs=$lte \
        local_train.batch_size=$batch_size \
        global_train.epochs=$global_epochs \
        logging.name_keys=[local_train.local_epochs,local_train.batch_size] \
        logging.constants=[large_encoder]
    echo "========================================"
    echo "========================================"
    echo "========================================"
done
