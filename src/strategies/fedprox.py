from flwr.server.strategy import FedProx


class FedProx(FedProx):

    def __init__(self, n_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_classes = n_classes
