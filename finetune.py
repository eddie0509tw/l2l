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
        self.weight_decay = cfg.get("weight_decay")
        if cfg.get("scheduler") is not None:
            s = cfg.get("scheduler")
            self.lr_step = s["milestones"]
            self.lr_factor = s["gamma"]

        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

        self.model = build_model(cfg)
        self.validator = build_validator(cfg)
        
        self.eval_step = cfg.get("eval_step", 1000)
        self.save_hyperparameters()

        self.init_wandb(cfg)

    def init_wandb(self, cfg):
        """Initializes Weights & Biases (wandb) using configuration settings."""
        wandb.init(
            project=cfg.get("project_name", "default"),
            entity=cfg.get("entity", None),
            dir=self.output_path,
            group=cfg.get("group", None),
            name=cfg.get("name", None),
            tags=cfg.get("tags", None),
            notes=cfg.get("notes", None),
            id=cfg.get("id", None),
            resume=cfg.get("resume", None)
        )

    def configure_optimizers(self):
        params = list(self.model.named_parameters())
        optimizer = torch.optim.AdamW(
            params, lr=self.lr, weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, self.lr_step, self.lr_factor)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def load_model_weight(self, model_weight_path):
        if os.path.exists(model_weight_path):
            state_dict = torch.load(model_weight_path)['state_dict']
            state_dict = {
                k.replace("model.", ""): v for k, v in state_dict.items()}

            self.model.load_state_dict(state_dict, strict=False)
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
            labels=labels)
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
        metrics = self.validator.get_metrics(preds, batch['labels'])

        return {"metrics": metrics, "loss": loss}

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
        self.validator.get_result()
        self.validator.init_metrics()


@hydra.main(config_path="conf", config_name="finetune", version_base="1.3")
def run(cfg: DictConfig):
    print(cfg)
    save_dir = (f"{cfg.dataset.name}_result/{cfg.task.name}/{cfg.model.name}")
    save_path = os.path.join(cfg.root_dir, save_dir)
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