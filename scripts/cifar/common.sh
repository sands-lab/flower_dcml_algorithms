export DATA_CONFIGURATIONS=(
    "iid_21clients_16seed_0.3test_0holdoutsize_400trainsize"
    "iid_21clients_16seed_0.3test_0holdoutsize_800trainsize"
    "iid_21clients_16seed_0.3test_0holdoutsize_1200trainsize"
    "iid_21clients_16seed_0.3test_0holdoutsize_1600trainsize"
    "iid_21clients_16seed_0.3test_0holdoutsize_2000trainsize"
)
export EPOCHS=(2)
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1

local_to_global_epochs_mapping() {
    case "$1" in
        1) global_epochs=160 ;;
        2) global_epochs=80 ;;
        4) global_epochs=40 ;;
        *) global_epochs=0 ;;  # Default value if input is not 1, 2, or 4
    esac
}

get_trainset_string () {
    number="${1##*_}"
    number="${number%trainsize}"
    export DATA_CONFIG_STRING="trainset${number}"
}

source ~/.venv/flower/bin/activate
