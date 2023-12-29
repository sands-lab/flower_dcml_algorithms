import json
import importlib

import torch


def construct_matrix(preds, targets, num_classes):
    assert num_classes == preds.shape[1]
    result_matrix = torch.zeros((num_classes, num_classes), device=preds.device)
    for y in torch.unique(targets):
        indices = (targets == y).nonzero(as_tuple=True)[0]
        result_matrix[y] = torch.sum(preds[indices].detach(), dim=0)
    return result_matrix


def init_model(client_capacity, n_classes, device, dataset):
    with open("model_mapping.json", "r") as fp:
        mapping = json.load(fp)
    class_string = mapping[dataset][str(client_capacity)]
    module_name, class_name = class_string.rsplit('.', 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    model = class_(n_classes)

    model.to(device)
    return model
