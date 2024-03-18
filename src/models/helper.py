import importlib

import torch

from src.helper.filepaths import FilePaths as FP
from src.helper.commons import read_json


def construct_matrix(preds, targets, num_classes):
    assert num_classes == preds.shape[1]
    result_matrix = torch.zeros((num_classes, num_classes), device=preds.device)
    for y in torch.unique(targets):
        indices = (targets == y).nonzero(as_tuple=True)[0]
        result_matrix[y] = torch.sum(preds[indices].detach(), dim=0)
    return result_matrix


def init_model_from_string(class_string, n_classes, ratio, device):
    module_name, class_name = class_string.rsplit('.', 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    model = class_(n_classes, ratio)
    model.to(device)
    return model


def init_model(client_capacity, n_classes, device, dataset):
    mapping = read_json(FP.GENERAL_MODEL_CONFIG, [])
    class_string = mapping[dataset][str(client_capacity)]
    model = init_model_from_string(class_string, n_classes, 1.0, device)
    return model


def init_pt_model(client_capacity, n_classes, device, dataset, rate):
    config = read_json(FP.PT_MODEL_CONFIG, [])

    class_string = config["base_models"][dataset]
    if rate is None:
        rate = config["capacity_mapping"][str(client_capacity)]

    model = init_model_from_string(class_string, n_classes, rate, device)
    return model


def simple_init_model_from_string(class_string, n_classes):
    module_name, class_name = class_string.rsplit('.', 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    model = class_(n_classes)
    return model
