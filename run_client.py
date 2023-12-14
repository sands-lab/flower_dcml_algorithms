import os
import tempfile

import flwr as fl
import hydra
from conf.config_schema import ParamConfig
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from dotenv import load_dotenv

from src.helper.commons import load_data_config


cs = ConfigStore.instance()
cs.store(name="config", node=ParamConfig)


@hydra.main(version_base=None, config_path="conf", config_name="base_config")
def main(cfg: ParamConfig):

    #read the environment variables
    data_home_folder = os.environ.get("FLTB_DATA_HOME_FOLDER")
    partitions_home_folder = "./data/partitions"
    client_idx = os.environ.get("FLTB_CLIENT_INDEX")
    server_ip = os.environ.get("FLTB_SERVER_IP")

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
