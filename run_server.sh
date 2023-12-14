#!/bin/bash

export LOG_TO_WANDB=1
export SYNC_WITH_WANDB_CLOUD=1
export IBEX_SIMULATION=0

# set up wandb to work offline
export HYDRA_FULL_ERROR=1

if [[ $LOG_TO_WANDB == 1 ]]; then
    echo "Logging to W&B"

    LOGS_DIR="./logs"
    TEMP_DIR=$(mktemp -d "$LOGS_DIR/wandb_XXXXXXX")
    echo "Temporary directory created for W&B offline logging: $TEMP_DIR"

    export WANDB_MODE=offline
    export WANDB_DIR=$TEMP_DIR
    export WANDB_RUN_ID=ray_wb_${WANDB_DIR: -7}

    echo "WANDB_MODE:        ${WANDB_MODE}"
    echo "WANDB_DIR:         ${WANDB_DIR}"
    echo "WANDB_RUN_ID:      ${WANDB_RUN_ID}"
fi

# activate the python environment
source /home/radovib/.venv/flower/bin/activate

echo "Server ::: Starting the server"
python run_server.py "$@"

echo "All clients completed!"

if [[ $LOG_TO_WANDB == 1 && $SYNC_WITH_WANDB_CLOUD == 1 ]]; then
    echo "Starting to sync offline results to Wandb cloud"
    source ./secrets.env
    cd logs/
    wandb online
    cd ..
    cd $TEMP_DIR
    wandb sync --include-offline --sync-all
else
    echo "Skipping upload to W&B cloud..."
fi
