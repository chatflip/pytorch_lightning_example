import os

import hydra
import pytorch_lightning as pl
import torch.optim as optim
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


class ImageSegmentator(pl.LightningModule):
    def __init__(self, args, model, criterions, metrics):
        super(ImageSegmentator, self).__init__()
        self.args = args
        self.model = model
        self.criterions = criterions
        self.metrics = metrics

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        image, target = batch
        output = self(image)
        jaccard_loss = self.criterions["jaccard_loss"](output, target)
        dice_loss = self.criterions["dice_loss"](output, target)
        lovasz_loss = self.criterions["lovasz_loss"](output, target)
        bce_loss = self.criterions["bce_loss"](output, target)
        loss = (
            0.25 * jaccard_loss
            + 0.25 * dice_loss
            + 0.25 * lovasz_loss
            + 0.25 * bce_loss
        )
        iou_acc = self.metrics(output, target)
        # GPU:0の結果のみlog保存
        self.log("train_jaccard_loss", jaccard_loss.item())
        self.log("train_dice_loss", dice_loss.item())
        self.log("train_lovasz_loss", lovasz_loss.item())
        self.log("train_bce_loss", bce_loss.item())
        self.log("train_loss", loss.item())
        self.log("train_iou", iou_acc.item())
        return {"loss": loss}

    def training_step_end(self, batch_parts):
        return batch_parts["loss"].mean()

    def validation_step(self, batch, batch_idx):
        image, target = batch
        output = self(image)
        jaccard_loss = self.criterions["jaccard_loss"](output, target)
        dice_loss = self.criterions["dice_loss"](output, target)
        lovasz_loss = self.criterions["lovasz_loss"](output, target)
        bce_loss = self.criterions["bce_loss"](output, target)
        loss = (
            0.25 * jaccard_loss
            + 0.25 * dice_loss
            + 0.25 * lovasz_loss
            + 0.25 * bce_loss
        )
        iou_acc = self.metrics(output, target)
        return (
            jaccard_loss.item(),
            dice_loss.item(),
            lovasz_loss.item(),
            bce_loss.item(),
            loss.item(),
            iou_acc.item(),
        )

    def validation_epoch_end(self, outputs):
        # TODO: drop_lastの場合計算合わない
        jaccard_loss_list = []
        dice_loss_list = []
        lovasz_loss_list = []
        bce_loss_list = []
        loss_list = []
        iou_acc_list = []
        for output in outputs:
            jaccard_loss_list.append(output[0].cpu().numpy())
            dice_loss_list.append(output[1].cpu().numpy())
            lovasz_loss_list.append(output[2].cpu().numpy())
            bce_loss_list.append(output[3].cpu().numpy())
            loss_list.append(output[4].cpu().numpy())
            iou_acc_list.append(output[5].cpu().numpy())
        jaccard_loss = sum(jaccard_loss_list) / len(jaccard_loss_list)
        dice_loss = sum(dice_loss_list) / len(dice_loss_list)
        lovasz_loss = sum(lovasz_loss_list) / len(lovasz_loss_list)
        bce_loss = sum(bce_loss_list) / len(bce_loss_list)
        loss = sum(loss_list) / len(loss_list)
        iou_acc = sum(iou_acc_list) / len(iou_acc_list)
        self.log("val_jaccard_loss", jaccard_loss)
        self.log("val_dice_loss", dice_loss)
        self.log("val_lovasz_loss", lovasz_loss)
        self.log("val_bce_loss", bce_loss)
        self.log("val_loss", loss)
        self.log("val_iou", iou_acc)

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
                T_0=num_training_samples,
                T_mult=1,
                eta_min=0.1 * self.args.lr,
                last_epoch=-1,
                verbose=True,
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
            dirpath=os.path.join(cwd, self.args.path2weight),
            filename=filename,
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")
        return [checkpoint_callback, lr_monitor]
