import os
import json
import tempfile
from functools import partial

import torch
import flwr as fl
import hydra
from hydra.core.config_store import ConfigStore, OmegaConf
from hydra.utils import instantiate
import wandb
from dotenv import load_dotenv

from conf.config_schema import ParamConfig
from src.helper.evaluation import WandbEvaluation
from src.helper.fl_helper import construct_config_fn
from src.helper.model_heterogeneity import inject_model_capacity, init_client_id_to_capacity_mapping
from src.helper.commons import set_seed, load_data_config, generate_wandb_config

cs = ConfigStore.instance()
cs.store(name="config", node=ParamConfig)


@hydra.main(version_base=None, config_path="conf", config_name="base_config")
def run(cfg: ParamConfig):
    print(cfg)

    data_home_folder = os.environ.get("FLTB_DATA_HOME_FOLDER")
    partitions_home_folder = "./data/partitions"
    partition_folder = \
        f"{partitions_home_folder}/{cfg.data.dataset}/{cfg.data.partitioning_configuration}"

    data_config = load_data_config(partition_folder)
    n_classes = {
        "cifar10": 10,
        "mnist": 10
    }[data_config["dataset_name"]]

    print(os.environ.get("LOG_TO_WANDB"))
    log_to_wandb = bool(int(os.environ.get("LOG_TO_WANDB")))
    print(log_to_wandb)
    fl_algorithm_name = cfg.fl_algorithm.strategy._target_.split(".")[-1]

    if log_to_wandb:
        print("Logging to wandb...")
        wandb_config_dict = generate_wandb_config(cfg) | data_config
        wandb.init(
            config=wandb_config_dict,
            name=fl_algorithm_name
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
        on_fit_config_fn=construct_config_fn(OmegaConf.to_container(cfg.local_train))
    )

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
            "experiment_folder": temp_dir
        }
        if strategy.__class__.__name__ in {"FedAvg", "FedProx"}:
            # these are the only strategies in which all the clients have the
            # very same model architecture the model capacity should be set in
            # the config file rather than being injected by the runtime
            client_fn_ = partial(client_fn, **common_kwargs)
        else:
            random_client_capacities = \
                init_client_id_to_capacity_mapping(data_config["n_clients"], 2)
            client_id_to_capacity_mapping_file = f"{temp_dir}/model_capacities.json"
            with open(client_id_to_capacity_mapping_file, "w") as fp:
                json.dump(random_client_capacities, fp)

            if strategy.__class__.__name__ in {"FedDF"}:
                strategy.set_client_capacity_mapping(client_id_to_capacity_mapping_file)
                strategy.set_dataset_name(data_config["dataset_name"])

            client_fn_ = partial(inject_model_capacity,
                                 client_fn=client_fn,
                                 client_capacities=random_client_capacities,
                                 **common_kwargs)

        fl.simulation.start_simulation(
            client_fn=client_fn_,
            num_clients=data_config["n_clients"],
            config=fl.server.ServerConfig(num_rounds=cfg.global_train.epochs),
            strategy=strategy,
            client_resources=client_resources,
        )

    if log_to_wandb:
        wandb.finish()


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    load_dotenv(override=True)
    load_dotenv("secrets.env", override=True)
    run()