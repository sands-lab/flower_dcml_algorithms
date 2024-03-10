#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=42
#SBATCH --gpus=2
#SBATCH --output=logs/txt/test_feddf_huge.txt

source ~/.venv/flower/bin/activate
source ./grid_search_scripts/common.sh

export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1

## grid search for FedAvg

lr=0.1
batch_size=12
slr=1e-3
weight_predictions=true

for lte in 2; do
    for ste in 40; do
        for kd_temperature in 0.1; do
            local_to_global_epochs_mapping $lte

            echo "$lr $local_epochs $batch_size"
            python fl.py fl_algorithm=feddf \
                local_train.lr=$lr \
                local_train.local_epochs=$lte \
                local_train.batch_size=$batch_size \
                global_train.epochs=$global_epochs \
                fl_algorithm.strategy.kd_lr=$slr \
                fl_algorithm.strategy.kd_epochs=$ste \
                fl_algorithm.strategy.kd_temperature=$kd_temperature \
                fl_algorithm.strategy.weight_predictions=$weight_predictions \
                logging.name_keys=[local_train.local_epochs,fl_algorithm.strategy.kd_epochs,fl_algorithm.strategy.weight_predictions,fl_algorithm.strategy.kd_temperature] \
                logging.constants=[test]
            echo "========================================"
            echo "========================================"
            echo "========================================"
        done
    done
done
