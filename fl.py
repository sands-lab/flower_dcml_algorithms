import os
import tempfile
from functools import partial

import flwr as fl
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
def run(cfg: ParamConfig):
    print(cfg)

    data_home_folder = os.environ.get("FLTB_DATA_HOME_FOLDER")
    partitions_home_folder = "./data/partitions"
    partition_folder = f"{partitions_home_folder}/{partition_folder}"

    data_config = load_data_config(partition_folder)
    n_classes = {
        "cifar10": 10,
        "mnist": 10
    }[data_config["dataset_name"]]

    wandb_config_dict = generate_wandb_config(cfg) | data_config
    # wandb.init(
    #     project="test-project",
    #     config=wandb_config_dict
    # )
    # evaluator = WandbEvaluation()

    # Create strategy
    strategy = instantiate(
        cfg.fl_algorithm.strategy,
        n_classes=n_classes,
        fraction_fit=cfg.global_train.fraction_fit,  # Sample 100% of available clients for training
        fraction_evaluate=cfg.global_train.fraction_eval,  # Sample 50% of available clients for evaluation
        min_fit_clients=1,  # Never sample less than 10 clients for training
        min_evaluate_clients=1,  # Never sample less than 5 clients for evaluation
        min_available_clients=1,  # Wait until all 10 clients are available
        # evaluate_metrics_aggregation_fn=evaluator.evaluate,
        on_fit_config_fn=construct_config_fn(cfg.local_train)
    )

    with tempfile.TemporaryDirectory(dir="data/client") as temp_dir:
        print(temp_dir)
        print(f"Temporary directory created: {temp_dir}")

        client_fn = instantiate(cfg.fl_algorithm.client, _partial_=True)
        client_resources = {"num_gpus": 1, "num_cpus": 1}
        client_fn_ = partial(client_fn,
                            images_folder=f"{data_home_folder}/{data_config['dataset_name']}",
                            partition_folder=partition_folder, seed=cfg.general.seed,
                            experiment_folder=temp_dir)

        set_seed(cfg.general.seed)
        fl.simulation.start_simulation(
            client_fn=client_fn_,
            num_clients=data_config["num_clients"],
            config=fl.server.ServerConfig(num_rounds=cfg.global_train.epochs),
            strategy=strategy,
            client_resources=client_resources,
        )
    # wandb.finish()


if __name__ == "__main__":
    load_dotenv()
    run()