import os
import json
import random
import pickle

import torch
import numpy as np


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


def generate_wandb_config(cfg):
    return {
        "algorithm": cfg.fl_algorithm.strategy._target_.split(".")[-1],
        "global_epochs": cfg.global_train.epochs,
        "fraction_fit": cfg.global_train.fraction_fit,
        "fraction_eval": cfg.global_train.fraction_eval,
        "lr": cfg.local_train.lr,
        "local_epochs": cfg.local_train.local_epochs,
        "batch_size": cfg.local_train.batch_size,
        "optimizer": cfg.local_train.optimizer,
        "seed": cfg.general.seed,
    }


def np_softmax(matrix: np.ndarray, axis: int):
    # using scipy implementation
    x_max = np.amax(matrix, axis=axis, keepdims=True)
    exp_x_shifted = np.exp(matrix - x_max)
    return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)


def get_numpy(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy()
