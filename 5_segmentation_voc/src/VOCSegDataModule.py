import os

import albumentations as A
import cv2
import hydra
import pytorch_lightning as L
import torch
from albumentations.pytorch import ToTensorV2

from .VOC2012Downloader import VOC2012Downloader
from .VOCSegDataset import VOCSegDataset


class VOCSegDataModule(L.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def prepare_data(self):
        VOC2012Downloader()

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
            dataset = VOCSegDataset(
                os.path.join(cwd, self.args.dataset_root),
                "train",
                self.args.num_classes,
                transform=self.train_transforms,
            )
        else:
            dataset = VOCSegDataset(
                os.path.join(cwd, self.args.dataset_root),
                "val",
                self.args.num_classes,
                transform=self.val_transforms,
            )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.args.arch.batch_size,
            shuffle=train,
            num_workers=os.cpu_count(),
            pin_memory=True,
            drop_last=train,
        )

    @property
    def train_transforms(self):
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    scale_limit=0.5,
                    rotate_limit=0,
                    shift_limit=0.1,
                    p=1,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                A.PadIfNeeded(
                    min_height=self.args.arch.image_height,
                    min_width=self.args.arch.image_width,
                    always_apply=True,
                    value=0,
                ),
                A.RandomCrop(
                    height=self.args.arch.image_height,
                    width=self.args.arch.image_width,
                    always_apply=True,
                ),
                A.GaussNoise(p=0.2),
                A.Perspective(p=0.5),
                A.OneOf(
                    [
                        A.CLAHE(p=1),
                        A.RandomBrightnessContrast(contrast_limit=0.0, p=1),
                        A.RandomGamma(p=1),
                    ],
                    p=0.9,
                ),
                A.OneOf(
                    [
                        A.Sharpen(p=1),
                        A.Blur(blur_limit=3, p=1),
                        A.MotionBlur(blur_limit=3, p=1),
                    ],
                    p=0.9,
                ),
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(brightness_limit=0.0, p=1),
                        A.HueSaturationValue(p=1),
                    ],
                    p=0.9,
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )

    @property
    def val_transforms(self):
        return A.Compose(
            [
                A.Resize(self.args.arch.image_height, self.args.arch.image_width),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )
