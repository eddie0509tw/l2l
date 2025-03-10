import torch
import os
from torch import nn
from transformers import DebertaConfig
from .deberta import DebertaForSequenceClassification

def build_model(cfg):
    model_name = cfg.model.get('model_name', None)
    name = cfg.dataset.get('name', None)
    task = cfg.task.get('name', None)
    if model_name:
        if model_name.find('deberta') != -1:
            cache_dir = os.path.join(cfg.dataset.root_dir, name, task, model_name)
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            model_config = DebertaConfig.from_pretrained(model_name, cache_dir=cache_dir)
            if name == 'glue' and task in ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']:
                model = DebertaForSequenceClassification(model_config)
    else:
        raise NotImplementedError("Model name not provided")
    
    return model