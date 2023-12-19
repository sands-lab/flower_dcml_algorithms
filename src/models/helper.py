import torch

from src.models.convnet import ConvNet, ConvNet2


def construct_matrix(preds, targets, num_classes):
    assert num_classes == preds.shape[1]
    result_matrix = torch.zeros((num_classes, num_classes), device=preds.device)
    for y in torch.unique(targets):
        indices = (targets == y).nonzero(as_tuple=True)[0]
        result_matrix[y] = torch.sum(preds[indices].detach(), dim=0)
    return result_matrix


def init_model(client_capacity, n_classes, device):
    model = {
        0: ConvNet,
        1: ConvNet2,
    }[client_capacity](n_classes)
    model.to(device)
    return model
