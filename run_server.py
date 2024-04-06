import os

import flwr as fl
from logging import INFO, WARNING
from flwr.common.logger import log
import hydra
from hydra.utils import instantiate
from hydra.core.config_store import OmegaConf
import wandb
from dotenv import load_dotenv

from src.helper.evaluation import WandbEvaluation
from src.helper.fl_helper import construct_config_fn
from src.helper.commons import set_seed, read_env_config
from src.helper.wandb import init_wandb
from src.fl.client_manager import HeterogeneousClientManager
from src.helper.environment_variables import EnvironmentVariables as EV


os.environ["HYDRA_FULL_ERROR"] = "1"

@hydra.main(version_base=None, config_path="config/hydra", config_name="base_config")
def main(cfg):

    server_ip = "0.0.0.0:8080"  # os.environ.get(EV.SERVER_ADDRESS)
    _, _, log_to_wandb, data_config, n_classes = read_env_config(cfg)
    log(INFO, f"Logging to wandb set to {log_to_wandb}")

    n_clients = int(os.getenv(EV.N_CLIENTS, 1))
    data_config["colext_job_id"] = os.getenv(EV.COLEXT_JOB_ID, "???")
    data_config["num_colext_clients"] = n_clients
    if log_to_wandb:
        init_wandb(cfg, data_config)
    evaluator = WandbEvaluation(log_to_wandb, patience=cfg.general.patience)

    set_seed(cfg.general.seed)
    print(f"Running server expecting {n_clients} clients...")
    strategy = instantiate(
        cfg.fl_algorithm.strategy,
        n_classes=n_classes,
        evaluation_freq=cfg.global_train.evaluation_freq,
        fraction_fit=cfg.global_train.fraction_fit,
        fraction_evaluate=cfg.global_train.fraction_eval,
        min_fit_clients=n_clients,
        min_evaluate_clients=n_clients,
        min_available_clients=n_clients,
        evaluate_metrics_aggregation_fn=evaluator.eval_aggregation_fn,
        fit_metrics_aggregation_fn=evaluator.fit_aggregation_fn,
        on_fit_config_fn=construct_config_fn(OmegaConf.to_container(cfg.local_train), evaluator)
    )
    strategy.set_dataset_name(data_config["dataset_name"])
    evaluator.set_strategy(strategy)

    log(INFO, f"Starting server on IP: {server_ip}")
    fl.server.start_server(
        server_address=server_ip,
        client_manager=HeterogeneousClientManager(n_clients),
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=cfg.global_train.epochs),
    )
    log(INFO, "Experiment completed.")
    if log_to_wandb:
        log(INFO, "Syncing wandb to local folder...")
        wandb.finish()
        log(INFO, "Wandb locally synced")


if __name__ == "__main__":
    if os.environ.get(EV.IBEX_SIMULATION, "0") != "0":
        log(WARNING, "Loading environment variables from `.env. This should only happen if you are running things in a simulation environment")
        load_dotenv()
    load_dotenv("secrets.env", override=True)
    main()
