#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=7:00:00
#SBATCH --cpus-per-task=42
#SBATCH --gpus=2
#SBATCH --output=logs/txt/grid_search_lg_fedavg_5mar.txt

source ~/.venv/flower/bin/activate
source ./grid_search_scripts/common.sh

export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1


## grid search for FedAvg
global_epochs=80
lte=2

for batch_size in 12 16; do
    for lr in 0.05 0.02; do

        echo "$lr $local_epochs $batch_size"
        python fl.py fl_algorithm=lg_fedavg \
            local_train.lr=$lr \
            local_train.local_epochs=$lte \
            local_train.batch_size=$batch_size \
            global_train.epochs=$global_epochs \
            logging.name_keys=[local_train.lr,local_train.batch_size] \
            logging.constants=[]
        echo "========================================"
        echo "========================================"
        echo "========================================"
    done

done
