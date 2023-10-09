import torch


def accuracy(preds, targets):
    return torch.sum(targets == preds) / preds.size()[0]