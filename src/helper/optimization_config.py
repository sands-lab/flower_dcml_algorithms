from dataclasses import dataclass, field

import torch


def init_optimizer(parameters, optimizer_name, lr, weight_decay):

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
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
    grad_norm_clipping_param: bool = 4.0
    weight_decay: float = 3e-4


    def __post_init__(self):
        self.model = self.model.train()
        self.model = self.model.to(self.device)
        if self.optimizer_name not in {"sgd", "adam"}:
            raise ValueError("Optimizer should be either `sgd` or `adam`")
        self.optimizer = init_optimizer(
            self.model.parameters(),
            self.optimizer_name,
            self.lr,
            self.weight_decay
        )
