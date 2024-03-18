from functools import partial

import wandb
import flwr as fl
import hydra
from hydra.core.config_store import OmegaConf
from dotenv import load_dotenv

from slower.simulation.app import start_simulation

from src.helper.commons import set_seed, generate_wandb_config
from src.helper.commons import read_env_config
from src.helper.evaluation import WandbEvaluation
from src.helper.model_heterogeneity import inject_client_capacity, init_client_id_to_capacity_mapping
from src.helper.fl_helper import construct_config_fn
from src.sl.sl_client import client_fn
from src.sl.sl_server_trainer import SlServerSegment
from src.sl.sl_strategy import SlStrategy
from src.helper.wandb import access_config


@hydra.main(version_base=None, config_path="config/hydra", config_name="base_config")
def run(cfg):
    print(cfg)

    data_home_folder, partition_folder, log_to_wandb, data_config, n_classes = read_env_config(cfg)

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
    evaluator = WandbEvaluation(log_to_wandb, cfg.general.patience)

    set_seed(cfg.general.seed)

    server_segment_fn_ = partial(SlServerSegment, dataset_name=data_config["dataset_name"], seed=cfg.general.seed, n_classes=n_classes)
    fit_config_fn = lambda _: cfg.local_train
    client_fn_ = partial(
        client_fn,
        images_folder=f"{data_home_folder}/{data_config['dataset_name']}",
        partition_folder=partition_folder,
        seed=cfg.general.seed,
        experiment_folder=None,
        separate_val_test_sets=cfg.general.separate_val_test_sets
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
    evaluator.set_strategy(strategy)

    random_client_capacities = \
            init_client_id_to_capacity_mapping(
                data_config["n_clients"],
                3,
                fixed_capacity=cfg.general.common_client_capacity
            )

    client_fn_ = partial(inject_client_capacity,
                         client_fn=client_fn_,
                         client_capacities=random_client_capacities)
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
