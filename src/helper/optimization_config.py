from dataclasses import dataclass, field

import torch


def init_optimizer(model, optimizer_name, lr):
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise RuntimeError(f"Optimizer {optimizer_name} is not allowed")
    return optimizer


@dataclass
class OptimizationConfig:
    model: torch.nn.Module
    dataloader: torch.utils.data.DataLoader
    lr: float
    epochs: int
    optimizer_name: str
    optimizer: torch.optim.Optimizer = field(init=False)
    device: torch.device


    def __post_init__(self):
        self.model = self.model.train()
        self.model = self.model.to(self.device)
        if self.optimizer_name not in {"sgd", "adam"}:
            raise ValueError("Optimizer should be either `sgd` or `adam`")
        self.optimizer = init_optimizer(self.model, self.optimizer_name, self.lr)
