import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.utils.data
# from torch import optim



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


import torch.nn.functional as F


def linear_combination(x, y, epsilon):
    return epsilon*x + (1-epsilon)*y
def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)