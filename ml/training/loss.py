import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=1)  # shape: (B, C, T) or (B, C)
        pt = torch.exp(logpt)

        # Gather log-probabilities at target positions
        logpt = logpt.gather(1, target.unsqueeze(1)).squeeze(1)  # shape: (B,)
        pt = pt.gather(1, target.unsqueeze(1)).squeeze(1)  # shape: (B,)

        # Optional class weights
        if self.weight is not None:
            at = self.weight.gather(0, target)
            logpt = logpt * at

        loss = -((1 - pt) ** self.gamma) * logpt

        # Mask ignore_index
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            loss = loss[mask]

        return loss.mean()
