#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --output=logs/txt/grid_search_fedavg_separate_large_local.txt

source ~/.venv/flower/bin/activate
source ./grid_search_scripts/common.sh

export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1

## grid search for FedAvg
# for local_epochs in 1 2 4; do
#     for lr in 0.02 0.05 0.1; do
#         for batch_size in 8 12 16; do
#             for model_capacity in 0 1 2; do
#                 local_to_global_epochs_mapping $local_epochs

#                 echo "$lr $local_epochs $batch_size"
#                 python fl.py fl_algorithm=fedavg \
#                     local_train.lr=$lr \
#                     local_train.local_epochs=$local_epochs \
#                     local_train.batch_size=$batch_size \
#                     global_train.epochs=$global_epochs \
#                     fl_algorithm.strategy.filter_capacity=$model_capacity \
#                     ray_client_resources.num_cpus=2 \
#                     ray_client_resources.num_gpus=0.14 \
#                     logging.constants=["separate"] \
#                     logging.name_keys=["local_train.local_epochs","local_train.lr","local_train.batch_size","fl_algorithm.strategy.filter_capacity"]
#                 echo "========================================"
#                 echo "========================================"
#                 echo "========================================"
#             done
#         done
#     done
# done

local_epochs=20
lr=0.1
batch_size=12

for model_capacity in 0 1 2; do
    local_to_global_epochs_mapping $local_epochs

    echo "$lr $local_epochs $batch_size"
    python fl.py fl_algorithm=fedavg \
        local_train.lr=$lr \
        local_train.local_epochs=$local_epochs \
        local_train.batch_size=$batch_size \
        global_train.epochs=24 \
        fl_algorithm.strategy.filter_capacity=$model_capacity \
        ray_client_resources.num_cpus=2 \
        ray_client_resources.num_gpus=0.14 \
        logging.constants=["separate"] \
        logging.name_keys=["local_train.local_epochs","local_train.lr","local_train.batch_size","fl_algorithm.strategy.filter_capacity"]
    echo "========================================"
    echo "========================================"
    echo "========================================"
done
