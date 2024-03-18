#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=28
#SBATCH --gpus=2
#SBATCH --output=logs/txt/cinic/federated_dropout.txt

source scripts/cinic/common.sh


for lte in "${EPOCHS[@]}"; do

    for data_config in "${DATA_CONFIGURATIONS[@]}"; do
        get_trainset_string $data_config

        python -u fl.py fl_algorithm=federated_dropout \
            local_train.local_epochs=$lte \
            global_train.epochs=$MAX_GLOBAL_EPOCHS \
            data.dataset=cinic \
            data.partitioning_configuration=$data_config \
            logging.constants=[$DATA_CONFIG_STRING] \
            logging.name_keys=[local_train.local_epochs]
    done
done
