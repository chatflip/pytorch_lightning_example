import os
import time

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from AnimeFaceDataModule import AnimeFaceDataModule
from CustomMlFlowLogger import CustomMlFlowLogger
from MlflowWriter import MlflowWriter
from model import mobilenet_v2
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import accuracy


class ImageClassifier(pl.LightningModule):
    def __init__(self, args, model, criterion):
        super(ImageClassifier, self).__init__()
        self.args = args
        self.model = model
        self.criterion = criterion

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        image, target = batch
        output = self(image)
        loss = self.criterion(output, target)
        if self.global_step % self.args.log_freq == 0:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            # GPU:0の結果のみlog保存
            self.log("train_loss", loss.item())
            self.log("train_acc1", acc1.item())
            self.log("train_acc5", acc5.item())
        return {"loss": loss}

    def training_step_end(self, batch_parts):
        return batch_parts["loss"].mean()

    def validation_step(self, batch, batch_idx):
        image, target = batch
        output = self(image)
        loss = self.criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        return loss.item(), acc1.item(), acc5.item()

    def validation_epoch_end(self, outputs):
        # TODO: drop_lastの場合計算合わない
        loss_list = []
        acc1_list = []
        acc5_list = []
        for output in outputs:
            loss_list.append(output[0].cpu().numpy())
            acc1_list.append(output[1].cpu().numpy())
            acc5_list.append(output[2].cpu().numpy())
        loss = sum(loss_list) / len(loss_list)
        acc1 = sum(acc1_list) / len(acc1_list)
        acc5 = sum(acc5_list) / len(acc5_list)
        self.log("val_loss", loss)
        self.log("val_acc1", acc1)
        self.log("val_acc5", acc5)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.args.optimizer.lr,
            momentum=self.args.optimizer.momentum,
            weight_decay=self.args.optimizer.weight_decay,
        )  # 最適化方法定義
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.args.optimizer.lr_step_size * len(self.train_dataloader()),
            gamma=self.args.optimizer.lr_gamma,
        )
        return [optimizer], [scheduler]

    def configure_callbacks(self):
        cwd = hydra.utils.get_original_cwd()
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            dirpath=os.path.join(cwd, self.args.path2weight),
            filename=f"{self.args.exp_name}_mobilenetv2_best",
        )
        return [checkpoint_callback]


def write_log_base(args, writer):
    for key in args:
        writer.log_param(key, args[key])
    writer.log_params_from_omegaconf_dict(args)
    writer.log_artifact(os.path.join(os.getcwd(), ".hydra/config.yaml"))
    writer.log_artifact(os.path.join(os.getcwd(), ".hydra/hydra.yaml"))
    writer.log_artifact(os.path.join(os.getcwd(), ".hydra/overrides.yaml"))
    return writer


@hydra.main(config_name="./../config/config.yaml")
def main(args):
    writer = MlflowWriter(args.exp_name)
    writer = write_log_base(args, writer)
    logger = CustomMlFlowLogger(writer)

    pl.seed_everything(args.seed)
    model = mobilenet_v2(pretrained=True, num_classes=args.num_classes)
    datamodule = AnimeFaceDataModule(args)
    criterion = nn.CrossEntropyLoss()
    plmodel = ImageClassifier(args, model, criterion)
    trainer = pl.Trainer(
        logger=logger,
        checkpoint_callback=False,
        gpus=2,
        max_epochs=args.epochs,
        flush_logs_every_n_steps=args.print_freq,
        log_every_n_steps=args.log_freq,
        accelerator="dp",
        precision=16 if args.apex else 32,
        deterministic=True,
        num_sanity_val_steps=-1,
    )
    starttime = time.time()  # 実行時間計測(実時間)
    trainer.fit(plmodel, datamodule=datamodule)
    trainer.test(plmodel, datamodule=datamodule, verbose=True)
    writer.move_mlruns()
    # 実行時間表示
    endtime = time.time()
    interval = endtime - starttime
    print(
        "elapsed time = {0:d}h {1:d}m {2:d}s".format(
            int(interval / 3600),
            int((interval % 3600) / 60),
            int((interval % 3600) % 60),
        )
    )


if __name__ == "__main__":
    main()
