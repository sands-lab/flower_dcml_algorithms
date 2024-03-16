import time

import pandas as pd
import numpy as np
import wandb


class WandbEvaluation:
    def __init__(self, log_to_wandb, patience) -> None:
        super().__init__()
        self.epoch = 0
        self.start_time = time.time()
        self.strategy = None
        self.accuracy_list = []
        self.patience = patience
        self.log_data = wandb.log if log_to_wandb else lambda *args: None

    def set_strategy(self, strategy):
        self.strategy = strategy

    def _init_dict(self, dataset_size, client_idx, type_, fit_aggregation):
        return {
            "dataset_size": dataset_size,
            "client_idx": client_idx,
            "type": type_,
            "epoch": self.epoch,
            "elapsed_time": time.time() - self.start_time,
            "fit_aggregation": fit_aggregation
        }

    def _evaluate(self, metrics, fit_aggregation):
        dataset_sizes = [n_examples for n_examples, _ in metrics]
        if len(metrics[0][1]) == 0:
            return {}
        metric_keys = {k for k in metrics[0][1].keys() if k not in {"client_id", "client_capacity"}}
        client_idxs = [metrics["client_id"] for _, metrics in metrics]
        client_capacities = [metrics["client_capacity"] for _, metrics in metrics]
        metric_values = {k: [vals[k] for _, vals in metrics] for k in metric_keys}

        pd_table = pd.DataFrame({
            "dataset_sizes": dataset_sizes,
            "client_idxs": client_idxs,
            "client_capacities": client_capacities,
            **metric_values
        })
        wandb_table = wandb.Table(dataframe=pd_table)

        global_log_dict = {
            "dataset_size": sum(dataset_sizes),
            "fl_epoch": self.epoch,
            "elapsed_time": time.time() - self.start_time,
            "fit_aggregation": fit_aggregation,
            "client_table": wandb_table
        }
        for k, v in metric_values.items():
            global_log_dict[k] = np.average(v, weights=dataset_sizes).item()
        self.log_data(global_log_dict)
        acc = np.average(metric_values["accuracy"], weights=dataset_sizes).item()
        self.accuracy_list.append(acc)
        print(acc)
        if (
            len(self.accuracy_list) > self.patience and
            self.accuracy_list[-self.patience-1] > max(self.accuracy_list[-self.patience:])
        ):
            t = "=" * 20
            print(f"\n{t}{t}ALGORITHM CONVERGED: {self.accuracy_list[-10:]}\n{t}{t}\n\n")
            self.strategy.set_converged()

        return {"accuracy": acc}

    def eval_aggregation_fn(self, metrics):
        return self._evaluate(metrics, False)

    def fit_aggregation_fn(self, metrics):
        return self._evaluate(metrics, True)
