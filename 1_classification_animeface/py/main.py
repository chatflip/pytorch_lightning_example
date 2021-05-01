import os
import time

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from AnimeFaceDataset import AnimeFaceDataset
from MlflowWriter import MlflowWriter
from model import mobilenet_v2
from utils import accuracy, get_worker_init


class ImageClassifier(pl.LightningModule):
    def __init__(self, args, model, writer):
        super(ImageClassifier, self).__init__()
        self.args = args
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.writer = writer

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        image, target = batch
        output = self(image)
        loss = self.criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        self.writer.log_metric("train/loss", loss.item())
        self.writer.log_metric("train/Acc1", acc1.item())
        self.writer.log_metric("train/Acc5", acc5.item())
        return loss

    def validation_step(self, batch, batch_idx):
        image, target = batch
        output = self(image)
        loss = self.criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        return loss.item(), acc1.item(), acc5.item()

    def validation_epoch_end(self, outputs):
        iteration = (self.current_epoch + 1) * len(self.train_dataloader())
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
        self.writer.log_metric("val/loss", loss, step=iteration)
        self.writer.log_metric("val/Acc1", acc1, step=iteration)
        self.writer.log_metric("val/Acc5", acc5, step=iteration)

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.args.optimizer.lr,
            momentum=self.args.optimizer.momentum,
            weight_decay=self.args.optimizer.weight_decay,
        )  # 最適化方法定義
        return optimizer

    def configure_callbacks(self):
        cwd = hydra.utils.get_original_cwd()
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            dirpath=os.path.join(cwd, self.args.path2weight),
            filename=f"{self.args.exp_name}_mobilenetv2_best",
        )
        return [checkpoint_callback]

    @property
    def normalize_transform(self):
        return transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    @property
    def train_transform(self):
        return transforms.Compose(
            [
                transforms.Resize(
                    self.args.image_size, InterpolationMode.BILINEAR
                ),  # リサイズ
                transforms.RandomCrop(self.args.crop_size),  # クロップ
                transforms.RandomHorizontalFlip(p=0.5),  # 左右反転
                transforms.ToTensor(),  # テンソル化
                self.normalize_transform,
            ]
        )

    @property
    def valid_transform(self):
        return transforms.Compose(
            [
                transforms.Resize(
                    self.args.image_size, InterpolationMode.BILINEAR
                ),  # リサイズ
                transforms.CenterCrop(self.args.crop_size),
                transforms.ToTensor(),  # テンソル化
                self.normalize_transform,
            ]
        )

    def __dataloader(self, train: bool):
        """Train/validation loaders."""
        cwd = hydra.utils.get_original_cwd()
        if train:
            dataset = AnimeFaceDataset(
                os.path.join(cwd, self.args.path2db, "train"), self.train_transform
            )
        else:
            dataset = AnimeFaceDataset(
                os.path.join(cwd, self.args.path2db, "val"), self.valid_transform
            )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            shuffle=train,
            num_workers=os.cpu_count(),
            pin_memory=True,
            drop_last=train,
            worker_init_fn=get_worker_init(self.args.seed),
        )

    def train_dataloader(self):
        return self.__dataloader(train=True)

    def val_dataloader(self):
        return self.__dataloader(train=False)


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

    pl.seed_everything(args.seed)
    model = mobilenet_v2(pretrained=True, num_classes=args.num_classes)
    plmodel = ImageClassifier(args, model, writer)

    trainer = pl.Trainer(
        gpus=2,
        accelerator="dp",
        progress_bar_refresh_rate=args.print_freq,
        precision=16,
        deterministic=True,
        max_epochs=args.epochs,
    )

    starttime = time.time()  # 実行時間計測(実時間)
    trainer.fit(plmodel)
    writer.set_terminated()
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
