import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import optim



def get_tensor_info(tensor):
    return f"Shape: {tensor.shape} | Type: {tensor.type()} | Device: {tensor.device}"



def get_loss_fn(loss_name):
    if loss_name=="mse":
        return torch.nn.MSELoss()
    elif loss_name=="bce":
        return torch.nn.BCELoss()
    elif loss_name=="bce_logit":
        return torch.nn.BCEWithLogitsLoss()
    elif loss_name=="ce":
        return torch.nn.CrossEntropyLoss()
    elif loss_name=="kl":
        return torch.nn.KLDivLoss()
    elif loss_name=="nll":
        return torch.nn.NLLLoss()
    else:
        assert False, f"loss_name {loss_name} not valid"