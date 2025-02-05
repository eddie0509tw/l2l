import torch
import numpy as np
from peft import get_peft_model

def parameter_cnt(model):
    return sum(p.numel() for p in model.parameters())

def trainable_parameter_cnt(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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

def get_lora_model(base_model, feature_extractor, lora_config):
    # Wrap model with LoRA
    peft_model = get_peft_model(base_model, lora_config)
    
    # Override forward
    original_forward = peft_model.forward

    def custom_forward(*args, **kwargs):
        # If 'input_ids' is there, remove it for Vision Transformers
        if "input_ids" in kwargs:
            print("Removing 'input_ids' from forward")
            kwargs.pop("input_ids")
        # The vision model expects pixel_values
        return original_forward(*args, **kwargs)

    peft_model.forward = custom_forward
    
    return peft_model, feature_extractor