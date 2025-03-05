import logging
import torch
from functools import partial

from transformers import PretrainedConfig
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def collate_fn(batch, max_seq_length=128):
    """Pads sequences to the specified max_seq_length."""
    input_ids = [torch.tensor(example["input_ids"][:max_seq_length]) for example in batch]
    attention_mask = [torch.tensor(example["attention_mask"][:max_seq_length]) for example in batch]
    labels = torch.tensor([example["label"] for example in batch])
    token_type_ids = [torch.tensor(example["token_type_ids"][:max_seq_length]) for example in batch]

    # Manually pad each sequence to `max_seq_length`
    def pad_tensor(tensor_list, pad_value=0):
        padded = torch.full((len(tensor_list), max_seq_length), pad_value, dtype=torch.long)
        for i, seq in enumerate(tensor_list):
            length = min(len(seq), max_seq_length)
            padded[i, :length] = seq[:length]  # Copy sequence into padded tensor
        return padded

    input_ids_padded = pad_tensor(input_ids, pad_value=0)
    attention_mask_padded = pad_tensor(attention_mask, pad_value=0)
    token_type_ids = pad_tensor(token_type_ids, pad_value=0)

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "labels": labels,
        'token_type_ids': token_type_ids
    }
    
def collate_fn_glue(cfg):
    return partial(collate_fn, max_seq_length=cfg['max_seq_length'])

def preprocess_glue(cfg, datasets, tokenizer, is_regression=False):

    name = cfg['dataset_name']
    if name != 'glue':
        raise NotImplementedError(f"Dataset {name} not supported or using the wrong processing function")
    task = cfg['task_name']
    # Labels
    if task is not None:
        is_regression = task == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Preprocessing the datasets
    if task is not None:
        sentence1_key, sentence2_key = task_to_keys[task]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if cfg.get('pad_to_max_length', None) and cfg['pad_to_max_length']:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    # if (
    #     model.cfg.label2id != Pretrainedcfg(num_labels=num_labels).label2id
    #     and task is not None
    #     and not is_regression
    # ):
    #     # Some have all caps in their cfg, some don't.
    #     label_name_to_id = {k.lower(): v for k, v in model.cfg.label2id.items()}
    #     if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
    #         label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
    #     else:
    #         logger.warning(
    #             "Your model seems to have been trained with labels, but they don't match the dataset: ",
    #             f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
    #             "\nIgnoring the model labels as a result.",
    #         )
    # elif task is None and not is_regression:
    #     label_to_id = {v: i for i, v in enumerate(label_list)}

    max_seq_length = cfg.get('max_seq_length', None)
    if max_seq_length is None:
        max_seq_length = tokenizer.model_max_length

    if max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(max_seq_length, tokenizer.model_max_length)
    return sentence1_key, sentence2_key, padding, max_seq_length, label_to_id