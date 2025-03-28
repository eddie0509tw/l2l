import torch
import numpy as np
import loratorch as lora
from peft import get_peft_model
from collections.abc import Mapping

def parameter_cnt(model):
    return sum(p.numel() for p in model.parameters())

def trainable_parameter_cnt(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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

def get_lora_model(base_model, lora_config):
    # Wrap model with LoRA
    peft_model = get_peft_model(base_model, lora_config)
    
    # # Override forward
    # original_forward = peft_model.forward

    # def custom_forward(*args, **kwargs):
    #     # If 'input_ids' is there, remove it for Vision Transformers
    #     if "input_ids" in kwargs:
    #         print("Removing 'input_ids' from forward")
    #         kwargs.pop("input_ids")
    #     # The vision model expects pixel_values
    #     return original_forward(*args, **kwargs)

    # peft_model.forward = custom_forward
    
    return peft_model


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

def clone_model_weight(source_model, target_model, debug=False):
    source_param = {name: param for name, param in source_model.named_parameters()}
    # Copy weights from pretrained model to local model
    for local_name, local_param in target_model.named_parameters():
        if local_name in source_param:  # Ensure shapes match
            try:
                param = source_param[local_name]
                local_param.data.copy_(param.data)
                if debug:
                    print(f"Param Copied {local_name}")  # Log successful copies
            except:
                raise ValueError(f"Param shape mismatch: {param.shape} != {local_name}/{local_param.shape}")
    source_buffers = {name: buffer for name, buffer in source_model.named_buffers()}
    # Copy buffers (like LayerNorm running stats)
    for local_name, local_buffer in target_model.named_buffers():
        if local_name in source_buffers:
            buffer = source_buffers[local_name]
            local_buffer.data.copy_(buffer.data)
            if debug:
                print(f"Buffer Copied {local_name}")
        else:
            raise ValueError(f"Buffer shape mismatch: {param.shape} != {local_name}/{local_param.shape}")


def nested_numpify(tensors):
    "Numpify `tensors` (even if it's a nested list/tuple/dict of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_numpify(t) for t in tensors)
    if isinstance(tensors, Mapping):
        return type(tensors)({k: nested_numpify(t) for k, t in tensors.items()})

    t = tensors.cpu()
    if t.dtype == torch.bfloat16:
        # As of Numpy 1.21.4, NumPy does not support bfloat16 (see
        # https://github.com/numpy/numpy/blob/a47ecdea856986cd60eabbd53265c2ca5916ad5d/doc/source/user/basics.types.rst ).
        # Until Numpy adds bfloat16, we must convert float32.
        t = t.to(torch.float32)
    return t.numpy()

import os
from transformers import AutoModel

def load_huggingface_model(model_weight_path, model_name=None, cache_dir=None):
    """
    Loads model weights from a specified path. 
    If the path does not exist, downloads the model to a cache directory.

    Args:
        model_weight_path (str): Path to the custom model weights.
        model_name (str): Pretrained model name or path (default: microsoft/deberta-base).
        cache_dir (str): Directory to store downloaded model weights.

    Returns:
        Loaded model instance.
    """

    os.makedirs(model_weight_path, exist_ok=True)

    if os.path.exists(os.path.join(model_weight_path, "pytorch_model.bin")):
        model = AutoModel.from_pretrained(model_weight_path)
    
    else:
        print(f"Warning: Model weights not found at {model_weight_path}")
        print(f"Downloading {model_name} to cache directory: {cache_dir}")

        model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)

        model.save_pretrained(model_weight_path)
        print(f"Model saved to {model_weight_path} for future use.")

    return model
