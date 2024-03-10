#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=42
#SBATCH --gpus=2
#SBATCH --output=logs/txt/cinic_private_training.txt

source ~/.venv/flower/bin/activate

export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1

local_epochs=1
lr=0.05
weight_decay=0.0001
batch_size=16

echo "$lr $local_epochs $batch_size"
python fl.py fl_algorithm=private_training \
    data.dataset=cinic \
    data.partitioning_configuration=iid_60clients_12seed_0.5test_0holdoutsize \
    global_train.epochs=100 \
    global_train.evaluation_freq=2 \
    local_train.lr=$lr \
    local_train.local_epochs=$local_epochs \
    local_train.batch_size=$batch_size \
    fl_algorithm.client.weight_decay=$weight_decay \
    logging.name_keys=[local_train.lr,local_train.batch_size,fl_algorithm.client.weight_decay] \
    logging.constants=[cinic]
echo "========================================"
echo "========================================"
