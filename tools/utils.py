import torch
import numpy as np


def parameter_cnt(model):
    return sum(p.numel() for p in model.parameters())

def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

def cmp_parameters(model1, model2, check_grad=False):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        # Compare parameter values
        if not torch.equal(p1, p2):
            return False
        
        # If check_grad is True, compare gradients as well.
        if check_grad:
            # Both gradients are None; consider them equal.
            if p1.grad is None and p2.grad is None:
                continue
            # If one gradient is None and the other is not, they differ.
            if (p1.grad is None) != (p2.grad is None):
                return False
            # Now both gradients are not None; compare them.
            if not torch.equal(p1.grad, p2.grad):
                return False
    return True
