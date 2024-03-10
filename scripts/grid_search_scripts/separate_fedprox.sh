#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --output=logs/txt/grid_search_fedprox_separate.txt

source ~/.venv/flower/bin/activate
source ./grid_search_scripts/common.sh

export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1

## grid search for FedProx
lr=0.05
batch_size=8

for local_epochs in 1 2 4; do
    for proximal_mu in 0.0001 0.0005 0.001; do
        for model_capacity in 0 1 2; do

            local_to_global_epochs_mapping $local_epochs

            echo "$lr $local_epochs $batch_size"
            python fl.py fl_algorithm=fedprox \
                local_train.lr=$lr \
                local_train.local_epochs=$local_epochs \
                local_train.batch_size=$batch_size \
                global_train.epochs=$global_epochs \
                fl_algorithm.strategy.filter_capacity=$model_capacity \
                fl_algorithm.strategy.proximal_mu=$proximal_mu \
                ray_client_resources.num_cpus=2 \
                ray_client_resources.num_gpus=0.14 \
                logging.constants=["separate"] \
                logging.name_keys=[local_train.local_epochs,fl_algorithm.strategy.filter_capacity,fl_algorithm.strategy.proximal_mu]
            echo "========================================"
            echo "========================================"
            echo "========================================"
        done
    done
done
