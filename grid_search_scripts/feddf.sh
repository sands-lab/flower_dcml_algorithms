#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=42
#SBATCH --gpus=2
#SBATCH --output=logs/txt/grid_search_feddf.txt

source ~/.venv/flower/bin/activate
source ./grid_search_scripts/common.sh

export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1

## grid search for FedAvg

lr=0.1
batch_size=12
slr=4e-4

for lte in 1 2 4; do
    for ste in 1; do
        for kd_temperature in 0.4 1.0; do
            for weight_predictions in true; do
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
                    logging.name_keys=[local_train.local_epochs,fl_algorithm.strategy.kd_epochs,fl_algorithm.strategy.weight_predictions,fl_algorithm.strategy.kd_temperature]
                echo "========================================"
                echo "========================================"
                echo "========================================"
            done
        done
    done
done
