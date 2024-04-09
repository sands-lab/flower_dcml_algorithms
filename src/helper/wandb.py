import wandb
from hydra.core.config_store import OmegaConf

from src.helper.commons import generate_wandb_config


def access_config(config, key_string):
    keys = key_string.split('.')  # Split the string by delimiter ('.')
    value = config
    for key in keys:
        value = value[key]  # Traverse the nested structure
    return value


def init_wandb(cfg, data_config):
    fl_algorithm_name = cfg.fl_algorithm.strategy._target_.split(".")[-1]

    extract = lambda k: k.split(".")[-1]
    constants = list(cfg.logging.constants)
    print(constants)
    wandb_name = "_".join(
        [fl_algorithm_name] +
        (constants if isinstance(constants, list) else [constants]) +
        [f"{extract(k)}{access_config(cfg, k)}" for k in cfg.logging.name_keys]
    )
    print("Logging to wandb...")
    wandb_config_dict = {**generate_wandb_config(OmegaConf.to_container(cfg)), **data_config}
    wandb.init(
        config=wandb_config_dict,
        name=wandb_name
    )
