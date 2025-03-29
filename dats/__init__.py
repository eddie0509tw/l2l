import os
from datasets import load_dataset
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer, DataCollatorWithPadding
from .glue import *
from torch.utils.data import DataLoader


def get_tokenizer(cfg, cache_dir=None):
    model_name = cfg.model.get('model_name', None)
    if model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=
            cache_dir if cache_dir is not None else None)
    else:
        raise NotImplementedError("Model name not provided")
    
    return tokenizer

def preprocess_function(
                        examples,
                        tokenizer,
                        sentence1_key,
                        sentence2_key,
                        padding,
                        max_seq_length,
                        label_to_id
                    ):
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
    meta_dataloader = {}
    name = cfg.dataset.name
    task = cfg.task.name
    if name == 'glue':
        cache_dir = os.path.join(cfg.dataset.root_dir, name, task)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        os.environ["HF_DATASETS_CACHE"] = cache_dir
        datasets = load_dataset(name, task, cache_dir=cache_dir)
        if cfg.dataset.get('use_tokenizer', None):
            tokenizer = get_tokenizer(cfg, cache_dir=cache_dir)
            sentence1_key, sentence2_key, padding, max_seq_length, label_to_id = preprocess_glue(
                                    cfg, datasets, tokenizer)

            datasets = datasets.map(lambda x: preprocess_function(
                                            x, tokenizer, sentence1_key, sentence2_key, \
                                            padding, max_seq_length, label_to_id
                                        ),
                                    batched=True, 
                                    load_from_cache_file=not cfg.dataset.get('overwrite_cache', False))
        columns_to_remove = []
        columns_to_remove.append('idx')
        if task == 'mrpc':
            columns_to_remove.append('sentence1')
            columns_to_remove.append('sentence2')
        else:
            columns_to_remove.append('sentence')

        datasets = datasets.remove_columns(columns_to_remove)
        train_dataset = datasets['train']
        val_dataset = datasets["validation_matched" if task == "mnli" else "validation"]
        test_dataset = datasets["test_matched" if task == "mnli" else "test"]
        if cfg.task.get('max_train_samples', None):
            train_dataset = train_dataset.select(range(cfg.task.max_train_samples))
        if cfg.get('max_val_samples', None):
            val_dataset = val_dataset.select(range(cfg.task.max_val_samples))
        if cfg.get('max_test_samples', None):
            test_dataset = test_dataset.select(range(cfg.task.max_test_samples))
            
        # data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers, 
            collate_fn=collate_fn_glue(cfg),
            shuffle=True)

        val_loader = DataLoader(
            val_dataset, 
            batch_size=cfg.batch_size, 
            num_workers=cfg.num_workers, 
            collate_fn=collate_fn_glue(cfg))

        test_loader = DataLoader(
            test_dataset, 
            batch_size=cfg.batch_size, 
            num_workers=cfg.num_workers, 
            collate_fn=collate_fn_glue(cfg))

        meta_dataloader.update({
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        })

    if len(meta_dataloader) < 1:
        raise NotImplementedError(f"Dataset {name} not supported")
    return meta_dataloader