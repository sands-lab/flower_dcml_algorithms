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


def _get_rng_file_names(_self):
    try:
        folder = _self.client_working_folder
        torch_file = f"{folder}/torch_rng.pth"
        np_file = f"{folder}/np_rng.pth"
        py_file = f"{folder}/py_rng.pth"
        return torch_file, np_file, py_file
    except AttributeError:
        return None, None, None


def sync_rng_state(func):
    def wrapper(self, *args, **kwargs):
        # define files
        torch_file, np_file, py_file = _get_rng_file_names(self)

        # load RNG states
        if torch_file is not None and os.path.exists(torch_file):
            torch.set_rng_state(torch.load(torch_file))
            with open(py_file, "rb") as fppy, open(np_file, "rb") as fpnp:
                random.setstate(pickle.load(fppy))
                np.random.set_state(pickle.load(fpnp))
        elif torch_file is not None:
            assert not os.path.exists(torch_file)
            assert not os.path.exists(np_file)
            assert not os.path.exists(py_file)

        # trigger computation
        result = func(self, *args, **kwargs)

        # store new RNG states
        if torch_file is None:
            torch_file, np_file, py_file = _get_rng_file_names(self)
            assert not (torch_file is None or np_file is None or py_file is None)

        torch.save(torch.get_rng_state(), torch_file)
        with open(py_file, "wb") as fppy, open(np_file, "wb") as fpnp:
            pickle.dump(np.random.get_state(), fpnp)
            pickle.dump(random.getstate(), fppy)

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
