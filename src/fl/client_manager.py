from flwr.common import GetPropertiesIns
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy


class HeterogeneousClientManager(SimpleClientManager):

    def __init__(self):
        super().__init__()
        self.client_to_capacity_mapping = {}

    def register(self, client: ClientProxy) -> bool:
        success = super().register(client)
        if success:
            ins = GetPropertiesIns({})
            props = client.get_properties(ins, None)
            client_capacity = props.properties["client_capacity"]
            self.client_to_capacity_mapping[client.cid] = client_capacity
            print(f"Registered client {client.cid} with capacity {client_capacity}")

        return success

    def unregister(self, client: ClientProxy) -> None:
        super().unregister(client)
        del self.client_to_capacity_mapping[client.cid]
