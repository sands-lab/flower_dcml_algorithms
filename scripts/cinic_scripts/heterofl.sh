#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=15:00:00
#SBATCH --cpus-per-task=28
#SBATCH --gpus=1
#SBATCH --output=logs/txt/cinic/heterofl_fraction_participation.txt

source ./cinic_scripts/common.sh

for lte in "${EPOCHS[@]}"; do
    local_to_global_epochs_mapping $lte
    for data_config in "${DATA_CONFIGURATIONS[@]}"; do
        get_trainset_string $data_config

        python -u fl.py fl_algorithm=heterofl \
            local_train.local_epochs=$lte \
            global_train.epochs=600 \
            data.dataset=cinic \
            data.partitioning_configuration=$data_config \
            logging.constants=[$DATA_CONFIG_STRING] \
            logging.name_keys=[local_train.local_epochs] \
            global_train.fraction_fit=0.22 \
            ray_client_resources.num_cpus=4 \
            ray_client_resources.num_gpus=0.2
    done
done
