from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from .glue import *
from torch.utils.data import DataLoader

def get_tokenizer(cfg):
    model_name = cfg.get('model_name', None)
    if model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        raise NotImplementedError("Model name not provided")
    
    return tokenizer

def preprocess_function(examples, tokenizer, sentence1_key, sentence2_key, padding, max_seq_length, label_to_id):
    # Tokenize the texts
    args = (
        (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

    # Map labels to IDs (not necessary for GLUE tasks)
    if label_to_id is not None and "label" in examples:
        result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        
    return result


def build_dataset(cfg):
    meta_data = {}
    name = cfg['dataset_name']
    task = cfg['task_name']
    if 'use_hf' in cfg and cfg['use_hf']:
        datasets = load_dataset(name, task)
        if name == 'glue':
            if cfg.get('use_tokenizer', None) and cfg['use_tokenizer']:
                tokenizer = get_tokenizer(cfg)
                sentence1_key, sentence2_key, padding, max_seq_length, label_to_id = preprocess_glue(
                                        cfg, datasets, tokenizer, is_regression=False)
                print(sentence1_key, sentence2_key, padding, max_seq_length, label_to_id)
                datasets = datasets.map(lambda x: preprocess_function(
                                            x, tokenizer, sentence1_key, sentence2_key, padding, max_seq_length, label_to_id),
                                        batched=True, 
                                        load_from_cache_file=not cfg.get('overwrite_cache', False))
            columns_to_remove = []
            if name == 'glue':
                columns_to_remove.append('idx')
                columns_to_remove.append('sentence')

            datasets = datasets.remove_columns(columns_to_remove)
            train_dataset = datasets['train']
            val_dataset = datasets["validation_matched" if task == "mnli" else "validation"]
            test_dataset = datasets["test_matched" if task == "mnli" else "test"]
            if cfg.get('max_train_samples', None) and cfg['max_train_samples'] > 0:
                train_dataset = train_dataset.select(range(cfg['max_train_samples']))
            if cfg.get('max_val_samples', None) and cfg['max_val_samples'] > 0:
                val_dataset = val_dataset.select(range(cfg['max_val_samples']))
            if cfg.get('max_test_samples', None) and cfg['max_test_samples'] > 0:
                test_dataset = test_dataset.select(range(cfg['max_test_samples']))
                
            # data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
            train_loader = DataLoader(
                train_dataset, batch_size=cfg['batch_size'], collate_fn=collate_fn_glue(cfg), shuffle=True)
            val_loader = DataLoader(
                val_dataset, batch_size=cfg['batch_size'], collate_fn=collate_fn_glue(cfg))
            test_loader = DataLoader(
                test_dataset, batch_size=cfg['batch_size'], collate_fn=collate_fn_glue(cfg))

            meta_data = {
                'train': train_loader,
                'val': val_loader,
                'test': test_loader
            }

    if len(meta_data) < 1:
        raise NotImplementedError(f"Dataset {name} not supported")
    return meta_data