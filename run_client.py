import os
import tempfile
import importlib

import flwr as fl
import hydra
from hydra.core.config_store import OmegaConf
from dotenv import load_dotenv

try:
    from colext import MonitorFlwrClient # type: ignore
except ModuleNotFoundError:
    print("Colext not found. Skipping...")
    MonitorFlwrClient = lambda cls: cls
from slower.client.app import start_client as sl_start_client

from src.helper.model_heterogeneity import get_client_capacity
from src.helper.commons import read_env_config
from src.helper.environment_variables import EnvironmentVariables as EV

os.environ["HYDRA_FULL_ERROR"] = "1"


@hydra.main(version_base=None, config_path="config/hydra", config_name="base_config")
def main(cfg):
    print(cfg.fl_algorithm)
    is_split_learning = "server_model" in cfg.fl_algorithm

    data_home_folder, partition_folder, _, _, _ = read_env_config(cfg)

    # read environment variables
    client_idx = os.environ.get(EV.CLIENT_ID)
    server_ip = os.environ.get(EV.SERVER_ADDRESS)
    print(client_idx)

    if cfg.general.common_client_capacity is None:
        client_capacity = get_client_capacity(client_idx, None)
    else:
        # only use this for FedAvg
        client_capacity = int(cfg.general.common_client_capacity)

    # Split the string to get the module name and class name
    client_init_kwargs = OmegaConf.to_container(cfg.fl_algorithm.client)
    client_class_str = client_init_kwargs.pop("_target_")
    module_name, class_name = client_class_str.rsplit(".", 1)
    module = importlib.import_module(module_name)

    # Get the class from the module
    client_class = getattr(module, class_name)
    client_class = MonitorFlwrClient(client_class)

    with tempfile.TemporaryDirectory(dir="data/client") as temp_dir:

        # instantiate the client
        client = client_class(
            cid=client_idx,
            client_capacity=client_capacity,
            images_folder=f"{data_home_folder}/{cfg.data.dataset}",
            partition_folder=partition_folder,
            seed=cfg.general.seed,
            experiment_folder=temp_dir,
            separate_val_test_sets=cfg.general.separate_val_test_sets,
            **client_init_kwargs
        )

        # start the client
        if is_split_learning:
            sl_start_client(server_address=server_ip, client=client.to_client())
        else:
            fl.client.start_client(server_address=server_ip, client=client.to_client())


if __name__ == "__main__":
    if os.environ.get(EV.IBEX_SIMULATION, "1") != "0":
        load_dotenv(override=True)
    main()
