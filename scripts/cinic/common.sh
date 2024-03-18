export DATA_CONFIGURATIONS=(
    "iid_24clients_19seed_0.15test_0.15val_0holdoutsize_500trainsize"
    "iid_24clients_19seed_0.15test_0.15val_0holdoutsize_1000trainsize"
    "iid_24clients_19seed_0.15test_0.15val_0holdoutsize_1500trainsize"
    "iid_24clients_19seed_0.15test_0.15val_0holdoutsize_2000trainsize"
    "iid_24clients_19seed_0.15test_0.15val_0holdoutsize_2500trainsize"
)
export EPOCHS=(2)
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export MAX_GLOBAL_EPOCHS=500

get_trainset_string () {
    number="${1##*_}"
    number="${number%trainsize}"
    export DATA_CONFIG_STRING="trainset${number}"
}

source ~/.venv/flower/bin/activate
