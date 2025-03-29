import os
import math
import torch
import hydra
import wandb
from omegaconf import DictConfig
from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from dats import build_dataset
from model import build_model
# from tools.vis import save_attention_loc
from tools.validator import build_validator

from peft import get_peft_model, LoraConfig

# Faster, but less precise
torch.set_float32_matmul_precision("high")
# sets seeds for numpy, torch and python.random.
seed_everything(42, workers=True)

# loss_dict = {
#     'CE' : nn.CrossEntropyLoss(),
#     'MSE': nn.MSELoss(),
#     'L1' : nn.L1Loss(),
#     'BCE': nn.BCELoss(),
#     'BCEWithLogits': nn.BCEWithLogitsLoss(),
#     'SmoothL1': nn.SmoothL1Loss(),
# }

class Finetuner(LightningModule):
    def __init__(self, cfg, output_path):
        super().__init__()

        self.batch_size = cfg.get("batch_size")

        self.lr = cfg.task.get("learning_rate")
        if not self.lr:
            if cfg.get("learning_rate") is not None:
                self.lr = cfg.get("learning_rate")
            else:
                raise ValueError(
                        f"Learning rate not found in config file. Please check your config file."
                    )

        self.weight_decay = cfg.get("weight_decay")
        if cfg.get("scheduler") is not None:
            s = cfg.get("scheduler")
            self.lr_step = s["milestones"]
            self.lr_factor = s["gamma"]

        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

        self.model = build_model(cfg)
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.shape)
        self.validator = build_validator(cfg)
        
        self.eval_step = cfg.get("eval_step", 1000)
        
        self.output_attentions = cfg.model.get("output_attentions", False)
        self.save_hyperparameters()
        
        if cfg.load_pretrained:
            model_weight_path = cfg.get("model_weight_path", None)
            use_hf_weights = cfg.model.get("use_hf_weights", False)
            model_name = cfg.model.get("model_name", None)

            if not model_weight_path and use_hf_weights:
                cache_dir = os.path.join(
                    cfg.dataset.root_dir, cfg.dataset.name, cfg.model.name, "weights")
                model_weight_path = cache_dir

            self.load_model_weight(
                model_weight_path, 
                use_hf_weights=use_hf_weights, 
                model_name=model_name)

        if cfg.model.get("apply_lora", False):
            self.init_lora(cfg)

        self.init_wandb(cfg)

    def init_wandb(self, cfg):
        """Initializes Weights & Biases (wandb) using configuration settings."""
        wandb.init(
            project=cfg.wandb.get("project", "default"),
            entity=cfg.wandb.get("entity", None),
            dir=self.output_path,
            group=cfg.wandb.get("group", None),
            name=cfg.wandb.get("name", None),
            tags=cfg.wandb.get("tags", None),
            notes=cfg.wandb.get("notes", None),
            id=cfg.wandb.get("id", None),
            resume=cfg.wandb.get("resume", None)
        )
    
    def init_lora(self, cfg):
        lora_meta_config = cfg.model.get("lora", None)
        lora_config = LoraConfig(**lora_meta_config)
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.shape)
        # exit()

    def configure_optimizers(self):
        params = list(self.model.named_parameters())
        optimizer = torch.optim.AdamW(
            params, lr=self.lr, weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, self.lr_step, self.lr_factor)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def load_model_weight(self, model_weight_path=None, **kwargs):
        print(f"Loading model weights from: {model_weight_path}")
        if model_weight_path is not None:
            if kwargs["use_hf_weights"]:
                if not os.path.exists(model_weight_path):
                    os.makedirs(model_weight_path, exist_ok=True)
                self.model.from_pretrained(
                    kwargs["model_name"], cache_dir=model_weight_path)

            elif os.path.exists(model_weight_path):
                state_dict = torch.load(model_weight_path)['state_dict']
                state_dict = {
                    k.replace("model.", ""): v for k, v in state_dict.items()}
                self.model.load_state_dict(state_dict, strict=False)

            else:
                raise FileNotFoundError(
                    f"Model wrong weigths path: {model_weight_path} or you can use HF weights")
        else:
            raise FileNotFoundError(
                f"Model weight file not found: {model_weight_path}")

    def forward(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        token_type_ids = batch['token_type_ids']

        outputs = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels,
            output_attentions=self.output_attentions)
        loss, logits, all_hidden_states, all_attentions = outputs
        # loss_dict = self.criterion(outputs, targets)

        # loss_value = loss_dict["total_loss"].detach()
        loss_value = loss.detach()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            raise Exception("Loss is not finite")

        return loss, logits, all_hidden_states, all_attentions

    def training_step(self, batch, batch_idx):
        loss, logits, all_hidden_states, all_attentions = self.forward(batch, batch_idx)

        log = {}
        log[f"train/loss"] = loss.detach()
        
        wandb.log({"train/loss": loss.item(), "step": self.global_step})

        self.log_dict(
            log,
            logger=True,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss, logits, all_hidden_states, all_attentions = self.forward(batch, batch_idx)
        log = {}
        log[f"val/loss"] = loss.detach()
        
        self.log_dict(
            log,
            logger=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size)

        preds = logits.detach()
        labels = batch['labels']
        results = self.validator.get_metrics(preds, labels)
        self.validator.update_metrics(results)

        return {"metrics": results, "loss": loss}

    def on_train_batch_end(self, out, batch, batch_idx):
        pass

    def on_validation_batch_end(self, out, batch, batch_idx):
        if batch_idx % self.eval_step == 0:
            metrics= out["metrics"]
            loss = out["loss"]
            for k, v in metrics.items():
                self.log(f"val/{k}", v)
                wandb.log({f"val/{k}": v, "step": self.global_step})

    def on_validation_epoch_end(self):
        self.validator.get_results()
        self.validator.init_metrics()


@hydra.main(config_path="conf", config_name="finetune", version_base="1.3")
def run(cfg: DictConfig):
    save_dir = (f"{cfg.dataset.name}_result/{cfg.task.name}/{cfg.model.name}")
    save_path = os.path.join(cfg.root_dir, save_dir)
    print(save_path)
    meta_dataloader = build_dataset(cfg)
    train_loader = meta_dataloader['train']
    val_loader = meta_dataloader['val']
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(save_path, "weight"),
        filename="best",
        monitor='val/loss',
        mode='min',
        save_top_k=1,
        save_last=True)
    callbacks = [lr_monitor, ckpt_cb]

    trainer = Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu",
                      devices=[cfg.gpu],
                      precision=16 if torch.cuda.is_available() else 32,
                      max_epochs=cfg.num_epochs,
                      gradient_clip_val=0.1,
                      deterministic=False,
                      num_sanity_val_steps=1,
                      callbacks=callbacks)

    module = Finetuner(cfg, save_path)
    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    run()