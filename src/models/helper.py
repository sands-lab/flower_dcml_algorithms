import torch

def construct_matrix(preds, targets, num_classes):
    result_matrix = torch.zeros((num_classes, preds.shape[1]), device=preds.device)  # Initialize result matrix with zeros
    for y in torch.unique(targets):
        indices = (targets == y).nonzero(as_tuple=True)[0]
        result_matrix[y] = torch.sum(preds[indices], dim=0)
    return result_matrix