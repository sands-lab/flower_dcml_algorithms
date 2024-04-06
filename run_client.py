import os
import tempfile

import flwr as fl
import hydra
from hydra.utils import instantiate
from dotenv import load_dotenv


from src.helper.model_heterogeneity import get_client_capacity
from src.helper.commons import read_env_config
from src.helper.environment_variables import EnvironmentVariables as EV

os.environ["HYDRA_FULL_ERROR"] = "1"

@hydra.main(version_base=None, config_path="config/hydra", config_name="base_config")
def main(cfg):

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

    client_fn = instantiate(cfg.fl_algorithm.client, _partial_=True)
    with tempfile.TemporaryDirectory(dir="data/client") as temp_dir:

        # instantiate the client
        client = client_fn(
            cid=client_idx,
            client_capacity=client_capacity,
            images_folder=f"{data_home_folder}/{cfg.data.dataset}",
            partition_folder=partition_folder,
            seed=cfg.general.seed,
            experiment_folder=temp_dir,
            separate_val_test_sets=cfg.general.separate_val_test_sets
        )

        # start the client
        fl.client.start_client(server_address=server_ip, client=client.to_client())


if __name__ == "__main__":
    if os.environ.get(EV.IBEX_SIMULATION, "1") != "0":
        load_dotenv()
    main()
