import numpy as np


def init_client_id_to_capacity_mapping(n_clients, n_capacities):
    capacities = np.tile(np.arange(n_capacities), n_clients // n_capacities)[:n_clients].tolist()
    # convert cid to string. This is either way dont by `json`, as it always converts keys to str
    return {
        str(cid): capacity for cid, capacity in zip(range(n_clients), capacities)
    }


def inject_model_capacity(cid, client_fn, client_capacities, **kwargs):
    return client_fn(client_capacity=client_capacities[cid], cid=int(cid), **kwargs)
