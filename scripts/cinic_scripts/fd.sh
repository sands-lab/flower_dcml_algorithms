#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=28
#SBATCH --gpus=2
#SBATCH --output=logs/txt/cinic/fd.txt

source ./cinic_scripts/common.sh

for lte in "${EPOCHS[@]}"; do
    local_to_global_epochs_mapping $lte
    for data_config in "${DATA_CONFIGURATIONS[@]}"; do
        get_trainset_string $data_config

        python -u fl.py fl_algorithm=fd \
            local_train.local_epochs=$lte \
            global_train.epochs=$global_epochs \
            data.dataset=cinic \
            data.partitioning_configuration=$data_config \
            logging.constants=[$DATA_CONFIG_STRING] \
            logging.name_keys=[local_train.local_epochs]
    done
done
