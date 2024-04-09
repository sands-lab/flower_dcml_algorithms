import os
import json
import random
import pickle

import torch
import numpy as np

from src.helper.filepaths import FilePaths as FP
from src.helper.environment_variables import EnvironmentVariables as EV


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_data_config(partition_folder):
    with open(f"{partition_folder}/generation_config.json", "r") as fp:
        data = json.load(fp)
    return data


def _get_rng_file_names(working_folder):
    try:
        torch_file = f"{working_folder}/torch_rng.pth"
        np_file = f"{working_folder}/np_rng.pth"
        py_file = f"{working_folder}/py_rng.pth"
        assert os.path.exists(torch_file) == \
                os.path.exists(np_file) == \
                os.path.exists(py_file)
        return torch_file, np_file, py_file
    except AttributeError:
        return None, None, None


def _save_rng_state(torch_file, np_file, py_file):
    torch.save(torch.get_rng_state(), torch_file)
    with open(py_file, "wb") as fppy, open(np_file, "wb") as fpnp:
        pickle.dump(np.random.get_state(), fpnp)
        pickle.dump(random.getstate(), fppy)


def save_rng_state_if_not_exists(working_folder):
    torch_file, np_file, py_file = _get_rng_file_names(working_folder)
    if not os.path.exists(torch_file):
        _save_rng_state(torch_file, np_file, py_file)


def sync_rng_state(func):
    def wrapper(_self, *args, **kwargs):
        # define files
        torch_file, np_file, py_file = \
            _get_rng_file_names(_self.client_working_folder)

        # load RNG states
        torch.set_rng_state(torch.load(torch_file))
        with open(py_file, "rb") as fppy, open(np_file, "rb") as fpnp:
            random.setstate(pickle.load(fppy))
            np.random.set_state(pickle.load(fpnp))

        # trigger computation
        result = func(_self, *args, **kwargs)

        _save_rng_state(torch_file, np_file, py_file)
        # return result
        return result
    return wrapper


def generate_wandb_config(dict_conf):

    algorithm = dict_conf["fl_algorithm"]["strategy"]["_target_"].split(".")[-1]
    relevant_keys = ["data", "general", "local_train", "global_train"]
    final_dict = {}
    final_dict["algorithm"] = algorithm

    for k in relevant_keys:
        for inner_key in dict_conf[k]:
            final_dict[f"{k}.{inner_key}"] = dict_conf[k][inner_key]
    for entity in ["client", "strategy"]:
        for k in dict_conf["fl_algorithm"][entity]:
            if k == "_target_":
                continue
            final_dict[f"{entity}.{k}"] = dict_conf["fl_algorithm"][entity][k]

    return final_dict


def np_softmax(matrix: np.ndarray, axis: int):
    # using scipy implementation
    x_max = np.amax(matrix, axis=axis, keepdims=True)
    exp_x_shifted = np.exp(matrix - x_max)
    return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)


def get_numpy(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy()


def read_json(file, key_list):
    with open(file, "r") as fp:
        value = json.load(fp)
    for key in key_list:
        value = value[key]
    return value


def read_env_config(cfg):
    data_home_folder = os.environ.get(EV.DATA_HOME_FOLDER)

    partitions_home_folder = "./data/partitions"
    partition_folder = \
        f"{partitions_home_folder}/{cfg.data.dataset}/{cfg.data.partitioning_configuration}"

    data_config = load_data_config(partition_folder)
    n_classes = read_json(FP.DATA_CONFIG, [data_config["dataset_name"], "n_classes"])

    log_to_wandb = bool(int(os.environ.get(EV.LOG_TO_WANDB, 0)))
    print(f"Logging to W&B set to: {log_to_wandb}")
    return data_home_folder, partition_folder, log_to_wandb, data_config, n_classes
