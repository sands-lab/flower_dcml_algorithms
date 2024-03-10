#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=18
#SBATCH --gpus=2
#SBATCH --output=logs/txt/cinic/fedavg_separate.txt

source ./cinic_scripts/common.sh

for lte in "${EPOCHS[@]}"; do
    local_to_global_epochs_mapping $lte
    for data_config in "${DATA_CONFIGURATIONS[@]}"; do
        for filter_capacity in 0 1 2; do
            get_trainset_string $data_config
            python -u fl.py fl_algorithm=fedavg \
                local_train.local_epochs=$lte \
                global_train.epochs=$global_epochs \
                data.dataset=cinic \
                data.partitioning_configuration=$data_config \
                fl_algorithm.strategy.filter_capacity=$filter_capacity \
                logging.constants=[$DATA_CONFIG_STRING] \
                logging.name_keys=[local_train.local_epochs,fl_algorithm.strategy.filter_capacity] \
                ray_client_resources.num_gpus=0.25
        done
    done
done
