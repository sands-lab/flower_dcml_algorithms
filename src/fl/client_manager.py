from flwr.common import GetPropertiesIns
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy


class HeterogeneousClientManager(SimpleClientManager):

    def __init__(self):
        super().__init__()
        self.client_to_capacity_mapping = {}

    def unregister(self, client: ClientProxy) -> None:
        super().unregister(client)
        print(f"Unregistering client {client.cid}")
        if client.cid in self.client_to_capacity_mapping:
            # this happens when unregistering a client for which we did not yet call get_props
            del self.client_to_capacity_mapping[client.cid]

    def sync_client_to_capacity(self, clients):
        # cannot use this in .register because the servicer did not start to listen to client
        # requests yet...
        for client_proxy in clients:
            cid = client_proxy.cid
            noins = GetPropertiesIns({})
            if cid not in self.client_to_capacity_mapping:
                res = client_proxy.get_properties(noins, None)
                capacity = res.properties["client_capacity"]
                self.client_to_capacity_mapping[cid] = capacity

    def sample(self, *args, **kwargs):
        clients = super().sample(*args, **kwargs)
        self.sync_client_to_capacity(clients)
        return clients

    def all(self):
        self.sync_client_to_capacity(self.clients.values())
        return self.clients
