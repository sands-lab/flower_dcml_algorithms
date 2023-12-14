import time

import wandb
import numpy as np

class WandbEvaluation:
    def __init__(self, log_to_wandb) -> None:
        super().__init__()
        self.epoch = 1
        self.start_time = time.time()
        self.log_data = wandb.log if log_to_wandb else lambda *args: None

    def evaluate(self, metrics):
        dataset_sizes = [n_examples for n_examples, _ in metrics]
        accuracies = [metrics["accuracy"] for _, metrics in metrics]
        client_idxs = [metrics["client_id"] for _, metrics in metrics]
        for dataset_size, accuracy, client_idx in zip(dataset_sizes, accuracies, client_idxs):
            self.log_data({
                "accuracy": accuracy,
                "dataset_size": dataset_size,
                "client_idx": client_idx,
                "type": "client",
                "epoch": self.epoch
            })
        self.log_data({
            "accuracy": np.mean(accuracies).item(),
            "dataset_size": sum(dataset_sizes),
            "client_idx": -1,
            "type": "global",
            "epoch": self.epoch,
            "elapsed_time": time.time() - self.start_time
        })
        self.epoch += 1

        return {"accuracy": np.mean(accuracies).item()}
