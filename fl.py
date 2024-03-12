import os
import json
import tempfile
from functools import partial

import torch
import flwr as fl
import hydra
from hydra.core.config_store import OmegaConf
from hydra.utils import instantiate
import wandb
from dotenv import load_dotenv

from src.helper.evaluation import WandbEvaluation
from src.helper.fl_helper import construct_config_fn
from src.helper.model_heterogeneity import inject_client_capacity, init_client_id_to_capacity_mapping
from src.helper.commons import set_seed, load_data_config, generate_wandb_config
from src.helper.commons import read_json
from src.fl.client_manager import HeterogeneousClientManager



def access_config(config, key_string):
    keys = key_string.split('.')  # Split the string by delimiter ('.')
    value = config
    for key in keys:
        value = value[key]  # Traverse the nested structure
    return value


@hydra.main(version_base=None, config_path="config/hydra", config_name="base_config")
def run(cfg):
    print(cfg)

    data_home_folder = os.environ.get("COLEXT_DATA_HOME_FOLDER")
    partitions_home_folder = "./data/partitions"
    partition_folder = \
        f"{partitions_home_folder}/{cfg.data.dataset}/{cfg.data.partitioning_configuration}"

    data_config = load_data_config(partition_folder)
    n_classes = read_json("config/data/data_configuration.json", [data_config["dataset_name"], "n_classes"])

    print(os.environ.get("LOG_TO_WANDB"))
    log_to_wandb = bool(int(os.environ.get("LOG_TO_WANDB")))
    print(log_to_wandb)
    fl_algorithm_name = cfg.fl_algorithm.strategy._target_.split(".")[-1]

    if log_to_wandb:
        extract = lambda k: k.split(".")[-1]
        constants = list(cfg.logging.constants)
        print(constants)
        wandb_name = "_".join(
            [fl_algorithm_name] +
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

    # Create strategy
    set_seed(cfg.general.seed)
    strategy = instantiate(
        cfg.fl_algorithm.strategy,
        n_classes=n_classes,
        evaluation_freq=cfg.global_train.evaluation_freq,
        fraction_fit=cfg.global_train.fraction_fit,
        fraction_evaluate=cfg.global_train.fraction_eval,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
        evaluate_metrics_aggregation_fn=evaluator.eval_aggregation_fn,
        fit_metrics_aggregation_fn=evaluator.fit_aggregation_fn,
        on_fit_config_fn=construct_config_fn(OmegaConf.to_container(cfg.local_train), evaluator)
    )
    evaluator.set_strategy(strategy)

    with tempfile.TemporaryDirectory(dir="data/client") as temp_dir:
        print(temp_dir)
        print(f"Temporary directory created: {temp_dir}")

        client_fn = instantiate(cfg.fl_algorithm.client, _partial_=True)

        client_resources = {
            "num_cpus": cfg.ray_client_resources.num_cpus,
            "num_gpus": cfg.ray_client_resources.num_gpus if torch.cuda.is_available() else 0
        }
        common_kwargs = {
            "images_folder": f"{data_home_folder}/{data_config['dataset_name']}",
            "partition_folder": partition_folder,
            "seed": cfg.general.seed,
            "experiment_folder": temp_dir,
            "separate_val_test_sets": cfg.general.separate_val_test_sets
        }
        random_client_capacities = \
            init_client_id_to_capacity_mapping(
                data_config["n_clients"],
                3,
                fixed_capacity=cfg.general.common_client_capacity
            )

        client_id_to_capacity_mapping_file = f"{temp_dir}/model_capacities.json"
        with open(client_id_to_capacity_mapping_file, "w") as fp:
            json.dump(random_client_capacities, fp)
        strategy.set_dataset_name(data_config["dataset_name"])

        client_fn_ = partial(inject_client_capacity,
                             client_fn=client_fn,
                             client_capacities=random_client_capacities,
                             **common_kwargs)

        fl.simulation.start_simulation(
            client_fn=lambda cid: client_fn_(cid).to_client(),
            num_clients=data_config["n_clients"],
            config=fl.server.ServerConfig(num_rounds=cfg.global_train.epochs),
            strategy=strategy,
            client_resources=client_resources,
            client_manager=HeterogeneousClientManager()
        )

    if log_to_wandb:
        wandb.finish()


if __name__ == "__main__":
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    load_dotenv(override=True)
    load_dotenv("secrets.env", override=True)
    run()
