from flwr.server.strategy import FedProx as FlFedProx


class FedProx(FlFedProx):

    def __init__(self, n_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_classes = n_classes
