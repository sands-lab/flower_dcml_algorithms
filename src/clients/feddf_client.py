from src.clients.fedprox_client import FedProxClient


class FedDFClient(FedProxClient):
    """
    In FedDF the client is oblivious to the mechanisms happening on the server. It just trains
    the model it receives with the same training process as FedProx.
    """


def client_fn(cid, **kwargs) -> FedDFClient:
    return FedDFClient(cid=int(cid), **kwargs)
