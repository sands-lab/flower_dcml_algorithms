#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=28
#SBATCH --gpus=2
#SBATCH --output=logs/txt/mnist/fedavg_common.txt


for common_capacity in 0 1 2; do

    python -u fl.py fl_algorithm=fedavg \
        global_train.epochs=800 \
        data.dataset=mnist \
        data.partitioning_configuration=iid_24clients_16seed_0.3test_0holdoutsize_1200trainsize \
        general.common_client_capacity=$common_capacity \
        logging.constants=[$DATA_CONFIG_STRING] \
        logging.name_keys=[local_train.local_epochs,general.common_client_capacity]
done
