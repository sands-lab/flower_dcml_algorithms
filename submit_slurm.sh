#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=1:40:00
#SBATCH --cpus-per-task=12
#SBATCH --gpus=1
#SBATCH --output=all_tests.txt

source ~/.venv/flower/bin/activate

export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1

python -u fl.py fl_algorithm=private_training
python -u fl.py fl_algorithm=fedavg
python -u fl.py fl_algorithm=fedprox
python -u fl.py fl_algorithm=feddf
python -u fl.py fl_algorithm=fd fl_algorithm.client.kd_weight=0.1

python fl.py fl_algorithm=ds_fl \
    fl_algorithm.strategy.public_dataset_name=cifar10 \
    fl_algorithm.strategy.public_dataset_size=2000 \
    fl_algorithm.strategy.aggregation_method=era \
    fl_algorithm.strategy.temperature=0.1
python fl.py fl_algorithm=ds_fl \
    fl_algorithm.strategy.public_dataset_name=cifar10 \
    fl_algorithm.strategy.public_dataset_size=2000 \
    fl_algorithm.strategy.aggregation_method=sa \
    fl_algorithm.strategy.temperature=-1
