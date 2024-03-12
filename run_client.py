import os
import tempfile

import flwr as fl
import hydra
from hydra.utils import instantiate
from dotenv import load_dotenv



@hydra.main(version_base=None, config_path="conf", config_name="base_config")
def main(cfg):

    #read the environment variables
    data_home_folder = os.environ.get("COLEXT_DATA_HOME_FOLDER")
    partitions_home_folder = "./data/partitions"
    client_idx = os.environ.get("COLEXT_CLIENT_ID")
    server_ip = os.environ.get("COLEXT_SERVER_ADDRESS")

    partitions_exp_folder = f"{partitions_home_folder}/{cfg.data.dataset}/{cfg.data.partitioning_configuration}"

    client_fn = instantiate(cfg.fl_algorithm.client, _partial_=True)
    with tempfile.TemporaryDirectory(dir="data/client") as temp_dir:

        # instantiate the client
        client = client_fn(
            cid=client_idx,
            images_folder=f"{data_home_folder}/{cfg.data.dataset}",
            partition_folder=partitions_exp_folder,
            seed=cfg.general.seed,
            experiment_folder=temp_dir
        )

        # start the client
        fl.client.start_numpy_client(server_address=server_ip, client=client)


if __name__ == "__main__":
    if os.environ.get("IBEX_SIMULATION", "1") != "0":
        load_dotenv()
    main()
