import numpy as np


def init_client_id_to_capacity_mapping(n_clients, n_capacities, fixed_capacity=None):
    if fixed_capacity is None:
        capacities = np.tile(np.arange(n_capacities), n_clients // n_capacities + 1)[:n_clients].tolist()
        # convert cid to string. This is either way dont by `json`, as it always converts keys to str
        return {
            str(cid): capacity for cid, capacity in zip(range(n_clients), capacities)
        }
    return {
        str(cid): fixed_capacity for cid in range(n_clients)
    }


def inject_model_capacity(cid, client_fn, client_capacities, **kwargs):
    return client_fn(client_capacity=client_capacities[cid], cid=int(cid), **kwargs)
