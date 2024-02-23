#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=15:00:00
#SBATCH --cpus-per-task=20
#SBATCH --gpus=1
#SBATCH --output=logs/txt/cinic_fedavg_common.txt

source ~/.venv/flower/bin/activate

export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1

## grid search for FedAvg
lr=0.05
batch_size=12

for local_epochs in 1 2; do
    for model_capacity in 0 1 2; do

        echo "$lr $local_epochs $batch_size"
        python fl.py fl_algorithm=fedavg \
            local_train.lr=$lr \
            local_train.local_epochs=$local_epochs \
            local_train.batch_size=$batch_size \
            global_train.epochs=500 \
            global_train.evaluation_freq=10 \
            global_train.fraction_fit=0.17 \
            data.dataset=cinic \
            data.partitioning_configuration=iid_60clients_12seed_0.5test_0holdoutsize \
            general.common_client_capacity=$model_capacity \
            fl_algorithm.strategy.filter_capacity=null \
            logging.constants=["cinic_common"] \
            logging.name_keys=[local_train.local_epochs,general.common_client_capacity]
        echo "========================================"
        echo "========================================"
        echo "========================================"
    done
done
