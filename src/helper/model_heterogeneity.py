import os

import numpy as np

from src.helper.commons import read_json
from src.helper.filepaths import FilePaths as FP
from src.helper.environment_variables import EnvironmentVariables as EV


def init_client_id_to_capacity_mapping(n_clients, n_capacities,
                                       fixed_capacity=None, lcc_perc=None, low_high_classes=None):
    assert (lcc_perc is None) == (low_high_classes is None)
    if fixed_capacity is None and lcc_perc is None:
        capacities = \
            np.tile(np.arange(n_capacities), n_clients // n_capacities + 1)[:n_clients].tolist()
        # convert cid to string. This is either way done by `json`, as it converts keys to str
        return {
            str(cid): capacity for cid, capacity in zip(range(n_clients), capacities)
        }
    elif fixed_capacity is not None:
        assert lcc_perc is None
        return {
            str(cid): fixed_capacity for cid in range(n_clients)
        }
    else:
        assert fixed_capacity is None
        assert len(low_high_classes) == 2
        assert low_high_classes[1] < low_high_classes[0]
        n_low_capacity_clients = int(n_clients * lcc_perc)
        capacities = [low_high_classes[0]] * n_low_capacity_clients + \
            [low_high_classes[1]] * (n_clients - n_low_capacity_clients)
        return {
            str(cid): capacity for cid, capacity in enumerate(capacities)
        }

def get_client_capacity(cid, client_capacities):
    if client_capacities is not None:
        capacity = client_capacities[cid]
    else:
        device_type = os.getenv(EV.DEVICE_TYPE)
        capacity = read_json(FP.DEVICE_TYPE_TO_CAPACITY_CONFIG, [device_type])
    return capacity


def inject_client_capacity(cid, client_fn, client_capacities=None, **kwargs):

    capacity = get_client_capacity(cid, client_capacities)
    return client_fn(client_capacity=capacity, cid=int(cid), **kwargs)
