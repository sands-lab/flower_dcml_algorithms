#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --output=logs/txt/cifar100/fedavg_separate.txt

source ./scripts/cifar100/common.sh

for lte in "${EPOCHS[@]}"; do
    for data_config in "${DATA_CONFIGURATIONS[@]}"; do
        for filter_capacity in 0 1 2; do
            get_trainset_string $data_config

            python -u fl.py fl_algorithm=fedavg \
                local_train.local_epochs=$lte \
                global_train.epochs=$MAX_GLOBAL_EPOCHS \
                data.dataset=cifar100 \
                data.partitioning_configuration=$data_config \
                logging.constants=[$DATA_CONFIG_STRING] \
                logging.name_keys=[local_train.local_epochs,fl_algorithm.strategy.filter_capacity] \
                ray_client_resources.num_gpus=0.14 \
                fl_algorithm.strategy.filter_capacity=$filter_capacity
        done
    done
done
