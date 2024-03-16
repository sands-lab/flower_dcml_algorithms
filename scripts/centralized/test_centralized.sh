#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=28
#SBATCH --gpus=1
#SBATCH --output=logs/txt/test_centralized.txt

source ../.venv/flower/bin/activate

MODEL="src.models.fully_conv_net.LargeAllConvNet"

if [ -f /data/interim/test_centralized.json ]; then
    rm data/interim/test_centralized.json
fi
echo "{}" > data/interim/test_centralized.json

for dataset_name in "cifar10" "cifar100" "cinic"; do
        partition_folder="./data/partitions/${dataset_name}/iid_20clients_10seed_0.25test_0.25val_0holdoutsize_1500trainsize"
        echo $partition_folder

        for i in $(seq 0 19); do
            echo "Filter client ${i}"
            python train_centralized.py \
                --partition_folder=$partition_folder \
                --model=$MODEL \
                --epochs=400 \
                --filter_client=$i \
                --patience=40
        done
        echo "Training centralized"
        for i in $(seq 0 4); do
            python train_centralized.py \
                    --partition_folder=$partition_folder \
                    --model=$MODEL \
                    --epochs=100 \
                    --patience=10 \
                    --seed=$i
        done
done
