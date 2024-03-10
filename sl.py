import os
from functools import partial

import wandb
import flwr as fl
import hydra
from hydra.core.config_store import OmegaConf
from dotenv import load_dotenv

from slower.simulation.app import start_simulation

from src.helper.commons import set_seed, load_data_config, generate_wandb_config
from src.helper.evaluation import WandbEvaluation
from src.helper.fl_helper import construct_config_fn
from src.sl.sl_client import client_fn
from src.sl.sl_server_trainer import SlServerSegment
from src.sl.sl_strategy import SlStrategy
from fl import access_config


@hydra.main(version_base=None, config_path="conf", config_name="base_config")
def run(cfg):
    print(cfg)

    data_home_folder = os.environ.get("FLTB_DATA_HOME_FOLDER")
    log_to_wandb = bool(int(os.environ.get("LOG_TO_WANDB")))

    partitions_home_folder = "./data/partitions"
    partition_folder = \
        f"{partitions_home_folder}/{cfg.data.dataset}/{cfg.data.partitioning_configuration}"

    data_config = load_data_config(partition_folder)
    n_classes = {
        "cifar10": 10,
        "mnist": 10,
        "cinic": 10
    }[data_config["dataset_name"]]

    if log_to_wandb:
        extract = lambda k: k.split(".")[-1]
        constants = list(cfg.logging.constants)
        wandb_name = "_".join(
            ["split_learning"] +
            (constants if isinstance(constants, list) else [constants]) +
            [f"{extract(k)}{access_config(cfg, k)}" for k in cfg.logging.name_keys]
        )
        print("Logging to wandb...")
        wandb_config_dict = generate_wandb_config(OmegaConf.to_container(cfg)) | data_config
        wandb.init(
            config=wandb_config_dict,
            name=wandb_name
        )
    evaluator = WandbEvaluation(log_to_wandb)

    set_seed(cfg.general.seed)

    server_segment_fn_ = partial(SlServerSegment, dataset_name=data_config["dataset_name"], seed=cfg.general.seed)
    fit_config_fn = lambda _: cfg.local_train
    client_fn_ = partial(
        client_fn,
        images_folder=f"{data_home_folder}/{data_config['dataset_name']}",
        partition_folder=partition_folder,
        seed=cfg.general.seed,
        experiment_folder=None,
    )
    strategy = SlStrategy(
        evaluation_freq=cfg.global_train.evaluation_freq,
        single_training_client=cfg.fl_algorithm.strategy.single_training_client,
        common_server=False,
        init_server_model_segment_fn=server_segment_fn_,
        config_server_segnent_fn=fit_config_fn,
        config_client_fit_fn=construct_config_fn(OmegaConf.to_container(cfg.local_train), evaluator),
        evaluate_metrics_aggregation_fn=evaluator.eval_aggregation_fn,
        fit_metrics_aggregation_fn=evaluator.fit_aggregation_fn
    )

    tmp = cfg.ray_client_resources
    start_simulation(
        client_fn=client_fn_,
        num_clients=data_config["n_clients"],
        strategy=strategy,
        client_resources=cfg.ray_client_resources,
        config=fl.server.ServerConfig(num_rounds=cfg.global_train.epochs)
    )

    if log_to_wandb:
        wandb.finish()


if __name__ == "__main__":
    load_dotenv(override=True)
    load_dotenv("secrets.env", override=True)
    run()
