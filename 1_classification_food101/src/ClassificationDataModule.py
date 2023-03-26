import os

import pytorch_lightning as pl
import torch
from Food101Dataset import Food101Dataset
from Food101Downloader import Food101Downloader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


class ClassificationDataModule(pl.LightningDataModule):
    def __init__(self, args, model_cfg):
        super().__init__()
        self.args = args
        model_input_size = model_cfg["input_size"]
        model_crop_cnt = model_cfg["crop_pct"]
        self.input_height = model_input_size[1]
        self.input_width = model_input_size[2]
        self.crop_height = int(round(self.input_height / model_crop_cnt))
        self.crop_width = int(round(self.input_width / model_crop_cnt))
        self.model_mean = model_cfg["mean"]
        self.model_std = model_cfg["std"]

    def prepare_data(self):
        Food101Downloader()

    def train_dataloader(self):
        return self.__dataloader(train=True)

    def val_dataloader(self):
        return self.__dataloader(train=False)

    def test_dataloader(self):
        return self.__dataloader(train=False)

    def __dataloader(self, train: bool):
        """Train/validation loaders."""
        if train:
            dataset = Food101Dataset(
                self.args.dataset_root,
                phase="train",
                transform=self.train_transforms,
            )
            if self.args.debug:
                dataset_length = len(dataset)
                dataset = torch.utils.data.Subset(
                    dataset, indices=list(range(dataset_length // 100))
                )
        else:
            dataset = Food101Dataset(
                self.args.dataset_root,
                phase="val",
                transform=self.val_transforms,
            )
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            shuffle=train,
            num_workers=os.cpu_count(),
            pin_memory=True,
            drop_last=train,
        )

    @property
    def train_transforms(self):
        return transforms.Compose(
            [
                transforms.TrivialAugmentWide(),
                transforms.Resize(
                    (self.crop_height, self.crop_width), InterpolationMode.BILINEAR
                ),  # リサイズ
                transforms.RandomCrop((self.input_height, self.input_width)),  # クロップ
                transforms.RandomHorizontalFlip(p=0.5),  # 左右反転
                transforms.ToTensor(),  # テンソル化
                transforms.Normalize(
                    mean=self.model_mean,
                    std=self.model_std,
                ),
            ]
        )

    @property
    def val_transforms(self):
        return transforms.Compose(
            [
                transforms.Resize(
                    (self.crop_height, self.crop_width), InterpolationMode.BILINEAR
                ),  # リサイズ
                transforms.CenterCrop((self.input_height, self.input_width)),
                transforms.ToTensor(),  # テンソル化
                transforms.Normalize(
                    mean=self.model_mean,
                    std=self.model_std,
                ),
            ]
        )
