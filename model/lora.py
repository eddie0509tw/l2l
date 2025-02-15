import torch
from torch import nn
from transformers import ViTConfig, ViTPreTrainedModel
from .vit import ViTModel
from typing import Optional, Tuple, Union
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from peft import LoraConfig



class ClassificationModel(ViTPreTrainedModel):
    def __init__(self, config: ViTConfig, peft_config: LoraConfig) -> None:
        super().__init__(config)
        if peft_config:
            assert peft_config.r > 0 and isinstance(peft_config.r, int), "LoRA requires r > 0"
            assert peft_config.lora_alpha > 0 and isinstance(peft_config.lora_alpha, int), "LoRA requires alpha > 0"
            config.use_peft = True
            config.peft_config = peft_config
        else:
            config.use_peft = False
            config.peft_config = None
        self.num_labels = config.num_labels
        self.vit = ViTModel(config)
        # Classifier head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.classifier(sequence_output[:, 0, :])

        if not return_dict:
            output = (logits,) + outputs[1:]
            return output
        sequence_output, pooled_output, hidden_states, attentions = outputs
        return logits, hidden_states, attentions
