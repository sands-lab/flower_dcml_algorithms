from dataclasses import dataclass


@dataclass
class PartitioningConfig:
    partitioning_method: str
    dataset_name: str
    n_clients: int
    seed: int
    holdout_set_size: int
    test_percentage: float
    raw_data_folder: str
    partitions_home_folder: str

    def get_config_str(self):
        common_config_lst = [
            self.partitioning_method,
            f"{self.n_clients}clients",
            f"{self.seed}seed",
            f"{self.test_percentage}test",
            f"{self.holdout_set_size}holdoutsize"
        ]
        common_config = "_".join(common_config_lst)
        return common_config

    def get_partition_folder(self):
        return f"{self.partitions_home_folder}/{self.dataset_name}/{self.get_config_str()}"


@dataclass
class DirichletPartitioning(PartitioningConfig):
    alpha: float
    min_size_of_dataset: int

    def get_config_str(self):
        common_config = super().get_config_str()
        return f"{common_config}_{self.alpha}alpha_{self.min_size_of_dataset}minsize"


@dataclass
class ShardsPartitioning(PartitioningConfig):
    n_shards: int

    def get_config_str(self):
        common_config = super().get_config_str()
        return f"{common_config}_{self.n_shards}shards"


@dataclass
class FDPartitioning(PartitioningConfig):
    pass


@dataclass
class IIDPartitioning(PartitioningConfig):
    pass
