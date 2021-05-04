import os

import hydra
import pytorch_lightning as pl
import torch.optim as optim
from pytorch_lightning.callbacks import ModelCheckpoint


class ImageSegmentator(pl.LightningModule):
    def __init__(self, args, model, criterion, metrics):
        super(ImageSegmentator, self).__init__()
        self.args = args
        self.model = model
        self.criterion = criterion
        self.metrics = metrics

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        image, target = batch
        output = self(image)
        loss = self.criterion(output, target)
        if self.global_step % self.args.log_freq == 0:
            iou_acc = self.metrics(output, target)
            # GPU:0の結果のみlog保存
            self.log("train_loss", loss.item())
            self.log("train_iou", iou_acc.item())
        return {"loss": loss}

    def training_step_end(self, batch_parts):
        return batch_parts["loss"].mean()

    def validation_step(self, batch, batch_idx):
        image, target = batch
        output = self(image)
        loss = self.criterion(output, target)
        iou_acc = self.metrics(output, target)
        return loss.item(), iou_acc.item()

    def validation_epoch_end(self, outputs):
        # TODO: drop_lastの場合計算合わない
        loss_list = []
        iou_acc_list = []
        for output in outputs:
            loss_list.append(output[0].cpu().numpy())
            iou_acc_list.append(output[1].cpu().numpy())
        loss = sum(loss_list) / len(loss_list)
        iou_acc = sum(iou_acc_list) / len(iou_acc_list)
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
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=num_training_samples,
            T_mult=1,
            eta_min=0.1 * self.args.lr,
            last_epoch=-1,
        )
        return [optimizer], [scheduler]

    def configure_callbacks(self):
        cwd = hydra.utils.get_original_cwd()
        filename = "{}_{}_{}_H{}_W{}.pth".format(
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
        return [checkpoint_callback]
