#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=28
#SBATCH --gpus=2
#SBATCH --output=logs/txt/cifar10/private_training.txt

source ./cifar_scripts/common.sh

for lte in "${EPOCHS[@]}"; do
    local_to_global_epochs_mapping $lte
    for data_config in "${DATA_CONFIGURATIONS[@]}"; do
        get_trainset_string $data_config

        python -u fl.py fl_algorithm=private_training \
            local_train.local_epochs=$lte \
            global_train.epochs=$global_epochs \
            data.dataset=cifar10 \
            data.partitioning_configuration=$data_config \
            logging.constants=[$DATA_CONFIG_STRING] \
            logging.name_keys=[local_train.local_epochs] \
            ray_client_resources.num_gpus=0.09
    done
done
