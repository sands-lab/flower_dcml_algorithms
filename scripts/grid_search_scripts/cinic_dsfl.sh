#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=42
#SBATCH --gpus=2
#SBATCH --output=logs/txt/cinic_dsfl.txt

source ~/.venv/flower/bin/activate
source ./grid_search_scripts/common.sh

export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1

lr=0.05
batch_size=12

echo "$lr $local_epochs $batch_size"
for local_epochs in 1 2 4; do
    local_to_global_epochs_mapping $local_epochs

    python fl.py fl_algorithm=ds_fl \
        data.dataset=cinic \
        data.partitioning_configuration=iid_60clients_12seed_0.5test_0holdoutsize \
        global_train.epochs=$global_epochs \
        global_train.evaluation_freq=4 \
        local_train.lr=$lr \
        local_train.local_epochs=$local_epochs \
        local_train.batch_size=$batch_size \
        fl_algorithm.strategy.public_dataset_size=1000 \
        fl_algorithm.client.kd_temperature=1.0 \
        logging.name_keys=[local_train.local_epochs,fl_algorithm.strategy.public_dataset_size,fl_algorithm.client.kd_temperature] \
        logging.constants=[cinic]
    echo "========================================"
    echo "========================================"

done
