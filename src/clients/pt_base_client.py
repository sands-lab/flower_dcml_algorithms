from src.clients.fedavg_client import FedAvgClient
from src.models.helper import init_pt_model


class PtClient(FedAvgClient):

    def _init_model(self):

        self.model = init_pt_model(
            client_capacity=self.client_capacity,
            n_classes=self.n_classes,
            device=self.device,
            dataset=self.dataset_name
        )
