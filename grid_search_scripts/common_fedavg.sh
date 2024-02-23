#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=42
#SBATCH --gpus=2
#SBATCH --output=logs/txt/grid_search_fedavg.txt

source ~/.venv/flower/bin/activate
source ./grid_search_scripts/common.sh

export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1

## grid search for FedAvg
lr=0.05
batch_size=12

for local_epochs in 1 2 4; do
    for model_capacity in 0 1 2; do
        local_to_global_epochs_mapping $local_epochs

        echo "$lr $local_epochs $batch_size"
        python fl.py fl_algorithm=fedavg \
            local_train.lr=$lr \
            local_train.local_epochs=$local_epochs \
            local_train.batch_size=$batch_size \
            global_train.epochs=$global_epochs \
            general.common_client_capacity=$model_capacity \
            fl_algorithm.strategy.filter_capacity=null \
            logging.constants=["common"] \
            logging.name_keys=[local_train.local_epochs,general.common_client_capacity]
        echo "========================================"
        echo "========================================"
        echo "========================================"
    done
done
