import torch
import numpy as np
import loratorch as lora
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

def get_lora_model_v2(module, lora_config, prefix=""):

    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        print(f"Layer: {full_name}, Type: {child.__class__.__name__}")

        # Print parameter shapes for this layer
        for param_name, param in child.named_parameters(recurse=False):
            print(f"    ├── Param: {param_name}, Shape: {param.shape}")

        # Recurse into submodules
        get_lora_model_v2(child, lora_config, prefix=full_name)


def clone_module(module, memo=None):
    """
    Clones a module while preserving gradients in PyTorch.
    Ensures that cloned parameters remain in the computational graph.
    """
    if memo is None:
        memo = {}

    if not isinstance(module, torch.nn.Module):
        return module

    clone = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = clone._parameters.copy()
    clone._buffers = clone._buffers.copy()
    clone._modules = clone._modules.copy()

    # Rewriting parameters (keeping computation graph)
    if hasattr(clone, '_parameters'):
        for param_key, param in module._parameters.items():
            if param is not None:
                param_ptr = param.data_ptr()
                if param_ptr in memo:
                    clone._parameters[param_key] = memo[param_ptr]
                else:
                    cloned = param.clone().requires_grad_(param.requires_grad)
                    clone._parameters[param_key] = cloned
                    memo[param_ptr] = cloned  # Store in memo

    # Handle buffers
    if hasattr(clone, '_buffers'):
        for buffer_key, buff in module._buffers.items():
            if buff is not None and buff.requires_grad:
                buff_ptr = buff.data_ptr()
                if buff_ptr in memo:
                    clone._buffers[buffer_key] = memo[buff_ptr]
                else:
                    cloned = buff.clone().requires_grad_(buff.requires_grad)
                    clone._buffers[buffer_key] = cloned
                    memo[buff_ptr] = cloned

    # Recursively clone submodules
    if hasattr(clone, '_modules'):
        for module_key, submodule in clone._modules.items():
            clone._modules[module_key] = clone_module(submodule, memo)

    # Ensure RNNs work properly (if applicable)
    if hasattr(clone, 'flatten_parameters'):
        clone = clone._apply(lambda x: x)

    if hasattr(module, "peft_config"):
        for peft_param_name, peft_param in module.named_parameters():
            if "lora_" in peft_param_name:
                cloned = peft_param.clone().requires_grad_(peft_param.requires_grad)
                clone._parameters[peft_param_name] = cloned
                # print(f"Cloned LoRA param {peft_param_name}: requires_grad={cloned.requires_grad}, grad_fn={cloned.grad_fn}")

    return clone

def clone(learner):
    """
    **Description**

    Returns a `MAML`-wrapped copy of the module whose parameters and buffers
    are `torch.clone`d from the original module.

    This implies that back-propagating losses on the cloned module will
    populate the buffers of the original module.
    For more information, refer to learn2learn.clone_module().

    **Arguments**

    * **first_order** (bool, *optional*, default=None) - Whether the clone uses first-
        or second-order updates. Defaults to self.first_order.
    * **allow_unused** (bool, *optional*, default=None) - Whether to allow differentiation
    of unused parameters. Defaults to self.allow_unused.
    * **allow_nograd** (bool, *optional*, default=False) - Whether to allow adaptation with
        parameters that have `requires_grad = False`. Defaults to self.allow_nograd.

    """
    from learn2learn.algorithms.maml import MAML
    first_order = learner.first_order
    allow_unused = learner.allow_unused
    allow_nograd = learner.allow_nograd
    lr = learner.lr

    return MAML(clone_module(learner.module),
                lr=lr,
                first_order=first_order,
                allow_unused=allow_unused,
                allow_nograd=allow_nograd)

def clone_model_weight(source_model, target_model):
    # Copy weights from pretrained model to local model
    for (name, param), (local_name, local_param) in zip(source_model.named_parameters(), target_model.named_parameters()):
        if param.shape == local_param.shape:  # Ensure shapes match
            local_param.data.copy_(param.data)
            print(f"Param Copied {name} -> {local_name}")  # Log successful copies
        else:
            raise ValueError(f"Param shape mismatch: {name}/{param.shape} != {local_name}/{local_param.shape}")
    # Copy buffers (like LayerNorm running stats)
    for (name, buffer), (local_name, local_buffer) in zip(source_model.named_buffers(), target_model.named_buffers()):
        if buffer.shape == local_buffer.shape:
            local_buffer.data.copy_(buffer.data)
            print(f"Copied buffer {name} -> {local_name}")
        else:
            raise ValueError(f"Buffer shape mismatch: {name}/{param.shape} != {local_name}/{local_param.shape}")
