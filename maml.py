#!/usr/bin/env python3

"""
Demonstrates how to:
    * use the MAML wrapper for fast-adaptation,
    * use the benchmark interface to load mini-ImageNet, and
    * sample tasks and split them in adaptation and evaluation sets.

To contrast the use of the benchmark interface with directly instantiating mini-ImageNet datasets and tasks, compare with `protonet_miniimagenet.py`.
"""

import random
import numpy as np
import os

import torch
from torch import nn, optim

import loratorch as lora
import learn2learn as l2l

from tools.utils import accuracy, parameter_cnt, trainable_parameter_cnt, get_lora_model, clone_model_weight
from transformers import AutoModelForImageClassification, ViTForImageClassification
from peft import LoraConfig
from model.lora import ClassificationModel

from dats.dataset import create_miniimgnat, create_omniglot
from torch.autograd import grad


def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device, use_custom=False):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)
    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots*ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]
    # Adapt the model
    logits = None
    for step in range(adaptation_steps):
        if not use_custom:
            logits = learner(adaptation_data).logits
        else:
            logits = learner(adaptation_data)[0]
        adaptation_error = loss(logits, adaptation_labels)

        learner.adapt(adaptation_error, allow_unused=True, allow_nograd=True)
        

    # Evaluate the adapted model
    if not use_custom:
        predictions = learner(evaluation_data).logits
    else:
        predictions = learner(evaluation_data)[0]
    evaluation_error = loss(predictions, evaluation_labels)
    evaluation_accuracy = accuracy(predictions, evaluation_labels)
    return evaluation_error, evaluation_accuracy

def eval(learner, meta_batch_size, adaptation_steps, shots, ways, tasksets, loss, device):
    meta_test_error = 0.0
    meta_test_accuracy = 0.0
    for task in range(meta_batch_size):
        batch = tasksets.test.sample()
        evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                           learner,
                                                           loss,
                                                           adaptation_steps,
                                                           shots,
                                                           ways,
                                                           device)
        meta_test_error += evaluation_error.item()
        meta_test_accuracy += evaluation_accuracy.item()
    return meta_test_error / meta_batch_size, meta_test_accuracy / meta_batch_size

def eval_huggingface(
        model,
        ways=5,
        shots=1,
        meta_lr=0.003,
        fast_lr=0.1,
        meta_batch_size=16,
        adaptation_steps=1,
        num_iterations=60000,
        cuda=True,
        seed=42,
        dataset='miniimagenet',):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda and torch.cuda.device_count():
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')

    if dataset == 'miniimagenet':
        tasksets = create_miniimgnat(train_samples=2*shots,
                                    train_ways=ways,
                                    test_samples=2*shots,
                                    test_ways=ways,)
    elif dataset == 'omniglot':
        tasksets = create_omniglot(train_ways=ways,
                                train_samples=2*shots,
                                test_ways=ways,
                                test_samples=2*shots,)
    else:
        raise ValueError("Unknown dataset.")

    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    loss = nn.CrossEntropyLoss(reduction='mean')
    return eval(maml, meta_batch_size, adaptation_steps, shots, ways, tasksets, loss, device)

def get_model(model_name='google/vit-base-patch16-224-in21k', ways=5, peft_config=None, use_custom=False):
    source_model = ViTForImageClassification.from_pretrained(model_name, ignore_mismatched_sizes=True, num_labels=ways)
    if use_custom:
        config = source_model.config
        model = ClassificationModel(config, peft_config=peft_config)
        clone_model_weight(source_model, model)
    else:
        model = source_model
    # Optionally disable memory efficient attention in model config if available.
    if hasattr(model.config, 'use_memory_efficient_attention'):
        model.config.use_memory_efficient_attention = False
    feature_extractor = None
    return model, feature_extractor

def main(
        model,
        ways=5,
        shots=1,
        meta_lr=0.001,
        fast_lr=0.1,
        meta_batch_size=32,
        adaptation_steps=1,
        num_iterations=1000,
        cuda=True,
        seed=42,
        save_dir = "./weights",
        use_custom=False,
        dataset='miniimagenet',
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda and torch.cuda.device_count():
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')
    
    if dataset == 'miniimagenet':
        tasksets = create_miniimgnat(train_samples=2*shots,
                                    train_ways=ways,
                                    test_samples=2*shots,
                                    test_ways=ways,)
    elif dataset == 'omniglot':
        tasksets = create_omniglot(train_ways=ways,
                                train_samples=2*shots,
                                test_ways=ways,
                                test_samples=2*shots,
                                num_tasks=20000,)

    params = parameter_cnt(model)
    print(f"Model has {params} parameters")

    # Optionally, print out trainable parameters:
    trainable_param = trainable_parameter_cnt(model)
    print("Trainable parameters:", trainable_param)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

    # Create model
    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    opt = optim.Adam(maml.parameters(), meta_lr)
    loss = nn.CrossEntropyLoss(reduction='mean')

    for iteration in range(num_iterations):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):

            for task in range(meta_batch_size):
                # Compute meta-training loss
                learner = maml.clone()
                batch = tasksets.train.sample()
                evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                                learner,
                                                                loss,
                                                                adaptation_steps,
                                                                shots,
                                                                ways,
                                                                device,
                                                                use_custom)
                evaluation_error.backward()
                meta_train_error += evaluation_error.item()
                meta_train_accuracy += evaluation_accuracy.item()

                # Compute meta-validation loss
                learner = maml.clone()
                batch = tasksets.validation.sample()
                evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                                learner,
                                                                loss,
                                                                adaptation_steps,
                                                                shots,
                                                                ways,
                                                                device,
                                                                use_custom)
                meta_valid_error += evaluation_error.item()
                meta_valid_accuracy += evaluation_accuracy.item()

            # Print some metrics
            print('\n')
            print('Iteration', iteration)
            print('Meta Train Error', meta_train_error / meta_batch_size)
            print('Meta Train Accuracy', meta_train_accuracy / meta_batch_size)
            print('Meta Valid Error', meta_valid_error / meta_batch_size)
            print('Meta Valid Accuracy', meta_valid_accuracy / meta_batch_size)

            # Average the accumulated gradients and optimize
            for p in maml.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(1.0 / meta_batch_size)
            opt.step()

    meta_test_error, meta_test_accuracy = eval(
        maml, meta_batch_size, adaptation_steps, shots, ways, tasksets, loss, device, use_custom)
    print('Meta Test Error', meta_test_error / meta_batch_size)
    print('Meta Test Accuracy', meta_test_accuracy / meta_batch_size)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    maml.module.save_pretrained(save_dir)
    # feature_extractor.save_pretrained(save_dir)
    print(f"Model and feature extractor saved to {save_dir}")

if __name__ == '__main__':
    n_ways = 5
    use_custom = True
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["query","value"],
        lora_dropout=0.0,
        bias="none",
        modules_to_save=["classifier"],
    )
    model, _ = get_model(use_custom=use_custom, peft_config=peft_config, ways=n_ways)
    main(model, dataset='miniimagenet',use_custom=use_custom, ways=n_ways)
    # meta_test_error, meta_test_accuracy = eval_huggingface(model)
    # print('Meta Test Error', meta_test_error)
    # print('Meta Test Accuracy', meta_test_accuracy)
