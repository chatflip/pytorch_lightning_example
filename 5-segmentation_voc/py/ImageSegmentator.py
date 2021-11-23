import os

import hydra
import pytorch_lightning as pl
import torch.optim as optim
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


class ImageSegmentator(pl.LightningModule):
    def __init__(self, args, model, criterions, criterions_weight, metrics):
        super(ImageSegmentator, self).__init__()
        self.args = args
        self.model = model
        self.criterions = criterions
        self.criterions_weight = criterions_weight
        self.metrics = metrics

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        image, target = batch
        output = self(image)
        loss = 0
        for key in self.criterions.keys():
            single_loss = self.criterions[key](output, target)
            loss += self.criterions_weight[key] * single_loss
            # GPU:0の結果のみlog保存
            self.log(f"train_{key}", single_loss.item())
        iou_acc = self.metrics(output, target)
        self.log("train_loss", loss.item())
        self.log("train_iou", iou_acc.item())
        return {"loss": loss}

    def training_step_end(self, batch_parts):
        return batch_parts["loss"].mean()

    def validation_step(self, batch, batch_idx):
        image, target = batch
        output = self(image)
        result_dict = {}
        loss = 0
        for key in self.criterions.keys():
            single_loss = self.criterions[key](output, target)
            loss += self.criterions_weight[key] * single_loss
            result_dict[key] = loss.item()
        iou_acc = self.metrics(output, target)
        result_dict["loss"] = loss.item()
        result_dict["iou"] = iou_acc.item()
        return result_dict

    def validation_epoch_end(self, outputs):
        # TODO: drop_last分のbatchsizeの考慮されてないから厳密ではない
        for key in outputs[0].keys():
            # validation_stepのdictのkeyごとに集計
            results = [outputs[i][key].cpu().numpy() for i in range(len(outputs))]
            self.log(f"val_{key}", sum(results) / len(results))

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        num_training_samples = len(self.trainer.datamodule.train_dataloader())
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.args.lr,
        )  # 最適化方法定義
        lr_scheduler = {
            "scheduler": optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=5 * num_training_samples,
                T_mult=2,
                eta_min=0.1 * self.args.lr,
                last_epoch=-1,
            ),
            "interval": "step",
            "name": "lr",
        }
        return [optimizer], [lr_scheduler]

    def configure_callbacks(self):
        cwd = hydra.utils.get_original_cwd()
        filename = "{}_{}_{}_H{}_W{}".format(
            self.args.exp_name,
            self.args.arch.decoder,
            self.args.arch.encoder,
            self.args.arch.image_height,
            self.args.arch.image_width,
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            dirpath=os.path.join(cwd, self.args.weight_root),
            filename=filename,
            save_top_k=1,
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")
        return [checkpoint_callback, lr_monitor]
