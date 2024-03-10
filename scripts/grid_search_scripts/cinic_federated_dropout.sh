#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=42
#SBATCH --gpus=2
#SBATCH --output=logs/txt/cinic_federated_dropout.txt

source ~/.venv/flower/bin/activate
source ./grid_search_scripts/common.sh

export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1

## grid search for FedAvg
lr=0.05
batch_size=12

for local_epochs in 1 2 4; do

    local_to_global_epochs_mapping $local_epochs

    echo "$lr $local_epochs $batch_size"
    python -u fl.py fl_algorithm=federated_dropout \
        data.dataset=cinic \
        data.partitioning_configuration=iid_60clients_12seed_0.5test_0holdoutsize \
        local_train.lr=$lr \
        local_train.local_epochs=$local_epochs \
        global_train.epochs=$global_epochs \
        local_train.batch_size=$batch_size \
        logging.name_keys=[local_train.lr,local_train.local_epochs,local_train.batch_size]
    echo "========================================"
    echo "========================================"
    echo "========================================"

done
