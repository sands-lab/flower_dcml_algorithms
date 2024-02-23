#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=24
#SBATCH --gpus=1
#SBATCH --output=logs/txt/cinic_heterofl.txt

source ~/.venv/flower/bin/activate

export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1

## grid search for FedAvg
batch_size=12
lr=0.05

for local_epochs in 1 2 4; do

    echo "$lr $local_epochs $batch_size"
    python fl.py fl_algorithm=heterofl \
        local_train.lr=$lr \
        data.dataset=cinic \
        data.partitioning_configuration=iid_60clients_12seed_0.5test_0holdoutsize \
        global_train.epochs=500 \
        global_train.evaluation_freq=10 \
        global_train.fraction_fit=0.17 \
        local_train.local_epochs=$local_epochs \
        local_train.batch_size=$batch_size \
        logging.name_keys=[local_train.lr,local_train.local_epochs,local_train.batch_size] \
        logging.constants=[cinic]
    echo "========================================"
    echo "========================================"
    echo "========================================"
done


# for whole_model in true false; do
#     for rate in 1.0 0.875 0.750 0.625; do
#     done
# done