#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=28
#SBATCH --gpus=2
#SBATCH --output=logs/txt/cinic/split_learning.txt

source scripts/cinic/common.sh
deactivate
source ../.venv/sl/bin/activate
PYTHONPATH=$PYTHONPATH:../slower


for lte in "${EPOCHS[@]}"; do

    for data_config in "${DATA_CONFIGURATIONS[@]}"; do
        get_trainset_string $data_config

        python -u sl.py fl_algorithm=split_learning \
            local_train.local_epochs=$lte \
            global_train.epochs=$MAX_GLOBAL_EPOCHS \
            data.dataset=cinic \
            data.partitioning_configuration=$data_config \
            logging.constants=[$DATA_CONFIG_STRING] \
            logging.name_keys=[local_train.local_epochs]
    done
done
