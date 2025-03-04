import os
import math
import torch
import hydra
from omegaconf import DictConfig
from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from datasets import build_dataset
from models import build_model, build_criterion, build_postprocessor
from tools.validator import build_validator
from tools.utils import box_cxcywh_to_xyxy
from tools.vis import save_attention_loc

# Faster, but less precise
torch.set_float32_matmul_precision("high")
# sets seeds for numpy, torch and python.random.
seed_everything(42, workers=True)

loss_dict = {
    'CE' : nn.CrossEntropyLoss(),
    'MSE': nn.MSELoss(),
    'L1' : nn.L1Loss(),
    'BCE': nn.BCELoss(),
    'BCEWithLogits': nn.BCEWithLogitsLoss(),
    'SmoothL1': nn.SmoothL1Loss(),
}


class Trainer(LightningModule):
    def __init__(self, cfg, output_path):
        super().__init__()

        self.batch_size = cfg.get("batch_size")
        self.lr = cfg.get("learning_rate")
        self.weight_decay = cfg.get("weight_decay")
        if cfg.get("scheduler") is not None:
            s = cfg.get("scheduler")
            self.lr_step = s["milestones"]
            self.lr_factor = s["gamma"]

        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

        self.model = build_model(cfg)
        self.criterion = loss_dict[cfg.get("loss")]
        self.postprocessor = build_postprocessor(cfg)
        self.validator = build_validator(cfg, "pose")
        self.validator.init_metrics()

        self.save_hyperparameters()

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
        samples, targets = batch

        outputs = self.model(samples)
        loss_dict = self.criterion(outputs, targets)

        loss_value = loss_dict["total_loss"].detach()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict)
            raise Exception("Loss is not finite")

        preds, pred_kpts, topk = \
            self.postprocessor(outputs, targets["ori_shape"])

        return loss_dict, preds, pred_kpts, topk

    def training_step(self, batch, batch_idx):
        loss_dict, preds, pred_kpts, topk = self.forward(batch, batch_idx)

        log = {}
        for key, value in loss_dict.items():
            log[f"train/{key}"] = value.detach()

        self.log_dict(
            log,
            logger=True,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size)

        return {"loss": loss_dict["total_loss"], "topk": topk}

    def validation_step(self, batch, batch_idx):
        loss_dict, preds, pred_kpts, topk = self.forward(batch, batch_idx)

        log = {}
        for key, value in loss_dict.items():
            log[f"val/{key}"] = value.detach()

        self.log_dict(
            log,
            logger=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size)

        _, targets = batch

        targs, tart_kpts = [], []
        for i in range(len(preds)):
            t_box = targets["boxes"][i]
            t_kpts = targets["keypoints"][i]
            t_label = targets["labels"][i]

            t_box, t_kpts, t_label = t_box.cpu(), t_kpts.cpu(), t_label.cpu()

            # convert to [x0, y0, x1, y1] format and to absolute coordinates
            # so that it matches the format of predicted boxes
            t_box = box_cxcywh_to_xyxy(t_box)
            img_h, img_w = targets["ori_shape"][i]
            scale_fct = torch.as_tensor([img_w, img_h, img_w, img_h])
            t_box = t_box * scale_fct[None, :]
            scale_fct = torch.as_tensor([img_w, img_h, 1])
            t_kpts = t_kpts * scale_fct[None, None, :]

            targs.append(torch.cat([t_box, t_label[:, None].float()], dim=-1))
            tart_kpts.append(t_kpts)

        self.validator.update_metrics(preds, targs, pred_kpts, tart_kpts)

        return {"preds": preds, "pred_kpts": pred_kpts, "topk": topk}

    def on_train_batch_end(self, out, batch, batch_idx):
        if batch_idx % 1000 == 0:
            prefix = '{}_{}'.format(
                os.path.join(self.output_path, 'train'), batch_idx)
            topk_box = out["topk"]["box"]
            topk_kpts = out["topk"]["kpts"]
            topk_loc = out["topk"]["loc"]
            topk_weight = out["topk"]["weight"]
            save_attention_loc(
                batch[0].image, topk_box, topk_kpts, topk_loc, topk_weight,
                '{}_attn_map.jpg'.format(prefix),
                num_display=self.batch_size
            )

    def on_validation_batch_end(self, out, batch, batch_idx):
        if batch_idx % 100 == 0:
            prefix = '{}_{}'.format(
                os.path.join(self.output_path, 'val'), batch_idx)
            topk_box = out["topk"]["box"]
            topk_kpts = out["topk"]["kpts"]
            topk_loc = out["topk"]["loc"]
            topk_weight = out["topk"]["weight"]
            save_attention_loc(
                batch[0].image, topk_box, topk_kpts, topk_loc, topk_weight,
                '{}_attn_map.jpg'.format(prefix),
                num_display=self.batch_size
            )

    def on_validation_epoch_end(self):
        self.validator.get_result()
        self.validator.init_metrics()


@hydra.main(config_path="conf", config_name="train", version_base="1.3")
def run(cfg: DictConfig):
    print(cfg)
    model_name = (f"{cfg.model.name}_{cfg.dataset.name}")
    save_path = os.path.join(cfg.output_dir, model_name)

    train_loader = build_dataset(cfg, "train")
    val_loader = build_dataset(cfg, "val")

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(save_path, "weight"),
        filename="best",
        monitor='val/total_loss',
        mode='min',
        save_top_k=1,
        save_last=True)
    callbacks = [lr_monitor, ckpt_cb]

    logger = TensorBoardLogger(
        save_dir=cfg.log_dir,
        name=model_name)

    trainer = Trainer(accelerator='gpu',
                      devices=[cfg.gpu],
                      precision=32,
                      max_epochs=cfg.num_epochs,
                      gradient_clip_val=0.1,
                      deterministic=False,
                      num_sanity_val_steps=1,
                      logger=logger,
                      callbacks=callbacks)

    module = Trainer(cfg, save_path)
    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    run()