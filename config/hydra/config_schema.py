from dataclasses import dataclass


@dataclass
class Data:
    dataset: str
    alpha: float

@dataclass
class Algorithm:
    name: str

@dataclass
class Train:
    epochs: int
    lr: float
    fraction_fit: float
    fraction_eval: float

@dataclass
class FederatedAlgorithm:
    name: str

@dataclass
class General:
    seed: int
    num_clients: int
    data: str

@dataclass
class ParamConfig:
    data: Data
    train: Train
    general: General
    fl_algorithm: FederatedAlgorithm