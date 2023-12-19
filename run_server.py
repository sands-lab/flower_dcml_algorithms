import os
import math

import flwr as fl
from logging import INFO, WARNING
from flwr.common.logger import log
import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
import wandb
from dotenv import load_dotenv

from conf.config_schema import ParamConfig
from src.helper.evaluation import WandbEvaluation
from src.helper.fl_helper import construct_config_fn
from src.helper.commons import set_seed, load_data_config, generate_wandb_config


cs = ConfigStore.instance()
cs.store(name="config", node=ParamConfig)


@hydra.main(version_base=None, config_path="conf", config_name="base_config")
def main(cfg: ParamConfig):

    partitions_home_folder = "./data/partitions"
    server_ip = os.environ.get("FLTB_SERVER_ADDRESS")
    log_to_wandb = int(os.environ.get("LOG_TO_WANDB", "0"))
    assert log_to_wandb in {0, 1}, "Logging to wandb should be set to either 0 or 1"
    log_to_wandb = bool(log_to_wandb)
    log(INFO, f"Logging to wandb set to {log_to_wandb}")

    partitions_exp_folder = f"{partitions_home_folder}/{cfg.data.dataset}/{cfg.data.partitioning_configuration}"

    data_config = load_data_config(partitions_exp_folder)
    assert data_config["dataset_name"] == cfg.data.dataset
    n_classes = {
        "cifar10": 10,
        "mnist": 10
    }[cfg.data.dataset]

    if log_to_wandb:
        log(INFO, "Initializing wandb")
        wandb_config_dict = {**generate_wandb_config(cfg), **data_config}
        wandb.init(
            project="test-project",
            config=wandb_config_dict
        )
    evaluator = WandbEvaluation(log_to_wandb)

    # Create strategy
    n_clients = int(data_config["num_clients"])
    log(INFO, f"Waiting for {n_clients} clients...")

    set_seed(cfg.general.seed)
    strategy = instantiate(
        cfg.fl_algorithm.strategy,
        n_classes=n_classes,
        fraction_fit=cfg.global_train.fraction_fit,
        fraction_evaluate=cfg.global_train.fraction_eval,
        min_fit_clients=int(math.floor(cfg.global_train.fraction_fit * n_clients)),
        min_evaluate_clients=int(math.floor(cfg.global_train.fraction_eval * n_clients)),
        min_available_clients=n_clients,
        evaluate_metrics_aggregation_fn=evaluator.evaluate,
        on_fit_config_fn=construct_config_fn(cfg.local_train)
    )

    log(INFO, f"Starting server on IP: {server_ip}")
    fl.server.start_server(
        server_address=server_ip,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=cfg.global_train.epochs)
    )
    log(INFO, "Experiment completed.")
    if log_to_wandb:
        log(INFO, "Syncing wandb to local folder...")
        wandb.finish()
        log(INFO, "Wandb locally synced")


if __name__ == "__main__":
    if os.environ.get("IBEX_SIMULATION", "0") != "0":
        log(WARNING, "Loading environment variables from `.env. This should only happen if you are running things in a simulation environment")
        load_dotenv()
    main()
