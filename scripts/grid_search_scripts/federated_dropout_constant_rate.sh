#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=15:00:00
#SBATCH --cpus-per-task=42
#SBATCH --gpus=2
#SBATCH --output=logs/txt/grid_search_federated_dropout_constant_rate.txt

source ~/.venv/flower/bin/activate
source ./grid_search_scripts/common.sh

export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1

## grid search for FedAvg

lr=0.05
batch_size=12
local_epochs=1

echo "$lr $local_epochs $batch_size"
python -u fl.py fl_algorithm=federated_dropout \
    local_train.lr=$lr \
    local_train.local_epochs=$local_epochs \
    global_train.epochs=500 \
    local_train.batch_size=$batch_size \
    general.common_client_capacity=2 \
    logging.constants=[constant_rate] \
    logging.name_keys=[local_train.lr,local_train.local_epochs,local_train.batch_size]
echo "========================================"
echo "========================================"
echo "========================================"

python -u fl.py fl_algorithm=federated_dropout \
    local_train.lr=$lr \
    local_train.local_epochs=$local_epochs \
    global_train.epochs=500 \
    local_train.batch_size=$batch_size \
    general.common_client_capacity=1 \
    logging.constants=[long] \
    logging.name_keys=[local_train.lr,local_train.local_epochs,local_train.batch_size]
echo "========================================"
echo "========================================"
echo "========================================"
