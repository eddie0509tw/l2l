import torch
from torch import nn
from transformers import DebertaConfig
from .deberta import DebertaForSequenceClassification

def build_model(cfg):
    model_name = cfg.get('model_name', None)
    name = cfg.get('dataset_name', None)
    task = cfg.get('task_name', None)
    if model_name:
        if model_name.find('deberta') != -1:
            model_config = DebertaConfig.from_pretrained(model_name)
            if name == 'glue' and task in ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']:
                model = DebertaForSequenceClassification(model_config)
    else:
        raise NotImplementedError("Model name not provided")
    
    return model