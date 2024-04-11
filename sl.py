import json
import tempfile
from functools import partial

import wandb
import flwr as fl
import hydra
from hydra.utils import instantiate

from hydra.core.config_store import OmegaConf
from dotenv import load_dotenv

from slower.simulation.app import start_simulation

from src.helper.commons import set_seed
from src.helper.commons import read_env_config
from src.helper.evaluation import WandbEvaluation
from src.helper.model_heterogeneity import inject_client_capacity, init_client_id_to_capacity_mapping
from src.helper.fl_helper import construct_config_fn
from src.helper.wandb import init_wandb


@hydra.main(version_base=None, config_path="config/hydra", config_name="base_config")
def run(cfg):
    print(cfg)

    data_home_folder, partition_folder, log_to_wandb, data_config, n_classes = read_env_config(cfg)

    if log_to_wandb:
        init_wandb(cfg, data_config)
    evaluator = WandbEvaluation(log_to_wandb, patience=cfg.general.patience)

    set_seed(cfg.general.seed)

    server_model_init_fn_ = instantiate(
        cfg.fl_algorithm.server_model,
        dataset_name=data_config["dataset_name"],
        seed=cfg.general.seed,
        n_classes=n_classes,
        _partial_=True
    )

    strategy = instantiate(
        cfg.fl_algorithm.strategy,
        n_classes=n_classes,
        evaluation_freq=cfg.global_train.evaluation_freq,
        init_server_model_fn=server_model_init_fn_,
        fraction_fit=cfg.global_train.fraction_fit,
        fraction_evaluate=cfg.global_train.fraction_eval,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
        evaluate_metrics_aggregation_fn=evaluator.eval_aggregation_fn,
        fit_metrics_aggregation_fn=evaluator.fit_aggregation_fn,
        # only these two functions differ from the flower API (plus provide common_server argument)
        config_client_fit_fn=construct_config_fn(OmegaConf.to_container(cfg.local_train), evaluator),
        config_server_segnent_fn=construct_config_fn(OmegaConf.to_container(cfg.local_train), None),
    )
    evaluator.set_strategy(strategy)

    with tempfile.TemporaryDirectory(dir="data/client") as temp_dir:
        client_fn = instantiate(cfg.fl_algorithm.client, _partial_=True)

        client_fn = partial(
            client_fn,
            images_folder=f"{data_home_folder}/{data_config['dataset_name']}",
            partition_folder=partition_folder,
            seed=cfg.general.seed,
            experiment_folder=temp_dir,
            separate_val_test_sets=cfg.general.separate_val_test_sets
        )

        random_client_capacities = \
            init_client_id_to_capacity_mapping(
                data_config["n_clients"],
                3,
                fixed_capacity=cfg.general.common_client_capacity,
                lcc_perc=cfg.general.lcc_perc,
                low_high_classes=cfg.general.low_high_classes,
            )
        print(json.dumps(random_client_capacities, indent=2))

        client_fn = partial(inject_client_capacity,
                            client_fn=client_fn,
                            client_capacities=random_client_capacities)
        data_config["n_clients"] = 1
        start_simulation(
            client_fn=client_fn,
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
