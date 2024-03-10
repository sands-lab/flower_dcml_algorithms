#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=42
#SBATCH --gpus=2
#SBATCH --output=logs/txt/test_fedkd_micro.txt

source ~/.venv/flower/bin/activate
source ./grid_search_scripts/common.sh

export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1

## grid search for FedAvg

lr=0.05
batch_size=8
kd_temperature=1.0

for lte in 4; do
    local_to_global_epochs_mapping $lte

    echo "$lr $batch_size"
    python fl.py fl_algorithm=fedkd \
        local_train.lr=$lr \
        local_train.local_epochs=$lte \
        local_train.batch_size=$batch_size \
        global_train.epochs=$global_epochs \
        fl_algorithm.client.temperature=$kd_temperature \
        logging.name_keys=[fl_algorithm.client.temperature,local_train.local_epochs] \
        logging.constants=[micro]
    echo "========================================"
    echo "========================================"
    echo "========================================"
done
