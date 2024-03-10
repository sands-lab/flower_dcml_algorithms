export DATA_CONFIGURATIONS=(
    "iid_24clients_16seed_0.2test_0holdoutsize_500trainsize"
    "iid_24clients_16seed_0.2test_0holdoutsize_1000trainsize"
    "iid_24clients_16seed_0.2test_0holdoutsize_1500trainsize"
    "iid_24clients_16seed_0.2test_0holdoutsize_2000trainsize"
    "iid_24clients_16seed_0.2test_0holdoutsize_2500trainsize"
    "iid_24clients_16seed_0.2test_0holdoutsize_3000trainsize"
)
export EPOCHS=(2 1)
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1

local_to_global_epochs_mapping() {
    case "$1" in
        1) global_epochs=240 ;;
        2) global_epochs=120 ;;
        4) global_epochs=60 ;;
        *) global_epochs=0 ;;  # Default value if input is not 1, 2, or 4
    esac
}

get_trainset_string () {
    number="${1##*_}"
    number="${number%trainsize}"
    export DATA_CONFIG_STRING="trainset${number}"
}

source ~/.venv/flower/bin/activate
