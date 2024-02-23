#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=24
#SBATCH --gpus=1
#SBATCH --output=logs/txt/cinic_feddf_2lte.txt

source ~/.venv/flower/bin/activate

export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1

## grid search for FedAvg

lr=0.1
batch_size=12
slr=4e-4
ste=1
kd_temperature=1.0

for lte in 2; do

    echo "$lr $local_epochs $batch_size"
    python -u fl.py fl_algorithm=feddf \
        data.dataset=cinic \
        data.partitioning_configuration=iid_60clients_12seed_0.5test_0holdoutsize \
        global_train.epochs=500 \
        global_train.evaluation_freq=10 \
        global_train.fraction_fit=0.17 \
        local_train.lr=$lr \
        local_train.local_epochs=$lte \
        local_train.batch_size=$batch_size \
        fl_algorithm.strategy.kd_lr=$slr \
        fl_algorithm.strategy.kd_epochs=$ste \
        fl_algorithm.strategy.kd_temperature=$kd_temperature \
        fl_algorithm.strategy.weight_predictions=$weight_predictions \
        logging.name_keys=[local_train.local_epochs,fl_algorithm.strategy.kd_epochs,fl_algorithm.strategy.weight_predictions,fl_algorithm.strategy.kd_temperature] \
        logging.constants=[cinic]

    echo "========================================"
    echo "========================================"
    echo "========================================"
done
