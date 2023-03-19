import os

import hydra
import pytorch_lightning as pl
import torch
from AnimeFaceDataset import AnimeFaceDataset
from AnimeFaceDownloader import AnimeFaceDownloader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode



class AnimeFaceDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def prepare_data(self):
        AnimeFaceDownloader()

    def train_dataloader(self):
        return self.__dataloader(train=True)

    def val_dataloader(self):
        return self.__dataloader(train=False)

    def test_dataloader(self):
        return self.__dataloader(train=False)

    def __dataloader(self, train: bool):
        """Train/validation loaders."""
        cwd = hydra.utils.get_original_cwd()
        if train:
            dataset = AnimeFaceDataset(
                os.path.join(cwd, self.args.dataset_root, "train"),
                self.train_transforms,
            )
        else:
            dataset = AnimeFaceDataset(
                os.path.join(cwd, self.args.dataset_root, "val"), self.val_transforms
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
                    self.args.image_size, InterpolationMode.BILINEAR
                ),  # リサイズ
                transforms.RandomCrop(self.args.crop_size),  # クロップ
                transforms.RandomHorizontalFlip(p=0.5),  # 左右反転
                transforms.ToTensor(),  # テンソル化
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    @property
    def val_transforms(self):
        return transforms.Compose(
            [
                transforms.Resize(
                    self.args.image_size, InterpolationMode.BILINEAR
                ),  # リサイズ
                transforms.CenterCrop(self.args.crop_size),
                transforms.ToTensor(),  # テンソル化
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
