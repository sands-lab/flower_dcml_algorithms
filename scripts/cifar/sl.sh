#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --output=logs/txt/cifar10/split_learning_additional_test.txt

source ./cifar_scripts/common.sh
export PYTHONPATH=$PYTHONPATH:../slower

# for lte in "${EPOCHS[@]}"; do
#     local_to_global_epochs_mapping $lte
#     for data_config in "${DATA_CONFIGURATIONS[@]}"; do
#         get_trainset_string $data_config

#         for single_client in true false; do
#             echo "Running $lte $data_config $single_client"
#             python -u sl.py fl_algorithm=split_learning \
#                 local_train.local_epochs=$lte \
#                 global_train.epochs=$global_epochs \
#                 data.dataset=cifar10 \
#                 data.partitioning_configuration=$data_config \
#                 logging.constants=[$DATA_CONFIG_STRING] \
#                 logging.name_keys=[fl_algorithm.strategy.single_training_client,local_train.local_epochs] \
#                 ray_client_resources.num_gpus=0.09 \
#                 fl_algorithm.strategy.single_training_client=$single_client
#         done
#     done
# done

global_epochs=200
data_config="iid_21clients_16seed_0.3test_0holdoutsize_400trainsize"
for lte in "${EPOCHS[@]}"; do
    get_trainset_string $data_config

    for single_client in false; do
        echo "Running $lte $data_config $single_client"
        python -u sl.py fl_algorithm=split_learning \
            local_train.local_epochs=$lte \
            global_train.epochs=$global_epochs \
            data.dataset=cifar10 \
            data.partitioning_configuration=$data_config \
            logging.constants=[$DATA_CONFIG_STRING] \
            logging.name_keys=[fl_algorithm.strategy.single_training_client,local_train.local_epochs] \
            ray_client_resources.num_gpus=0.09 \
            fl_algorithm.strategy.single_training_client=$single_client
    done
done
