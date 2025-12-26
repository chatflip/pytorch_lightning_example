import os

import albumentations as A
import cv2
import hydra
import pytorch_lightning as L
import torch
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .VOC2012Downloader import VOC2012Downloader
from .VOCSegDataset import VOCSegDataset


class VOCSegDataModule(L.LightningDataModule):
    """VOCセグメンテーションタスク用のPyTorch Lightning DataModule

    このモジュールはVOC2012セグメンテーションデータセットの
    データ準備、読み込み、変換を処理する。
    """

    def __init__(self, args: DictConfig) -> None:
        """VOCSegDataModuleを初期化する

        Args:
            args: dataset_root、num_classes、arch設定などを含む設定引数。
        """
        super().__init__()
        self.args = args

    def prepare_data(self) -> None:
        """データセットをダウンロードして準備する

        このメソッドは単一のGPU/プロセスからのみ呼び出される。
        """
        VOC2012Downloader()

    def train_dataloader(self) -> DataLoader:
        """訓練用データローダーを作成して返す

        Returns:
            DataLoader: 訓練用データローダー。
        """
        return self.__dataloader(train=True)

    def val_dataloader(self) -> DataLoader:
        """検証用データローダーを作成して返す

        Returns:
            DataLoader: 検証用データローダー。
        """
        return self.__dataloader(train=False)

    def test_dataloader(self) -> DataLoader:
        """テスト用データローダーを作成して返す

        Returns:
            DataLoader: テスト用データローダー。
        """
        return self.__dataloader(train=False)

    def __dataloader(self, train: bool) -> DataLoader:
        """訓練/検証用データローダーを作成する

        Args:
            train: Trueの場合、訓練用データローダーを作成し、それ以外は検証/テスト用。

        Returns:
            DataLoader: 指定されたフェーズ用に設定されたデータローダー。
        """
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
    def train_transforms(self) -> A.Compose:
        """訓練用データ変換を取得する

        Returns:
            A.Compose: HorizontalFlip、ShiftScaleRotate、RandomCrop、
                正規化などの拡張技術を含む訓練用変換の合成。
        """
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
    def val_transforms(self) -> A.Compose:
        """検証/テスト用データ変換を取得する

        Returns:
            A.Compose: Resize、Normalize、ToTensorV2を含む検証用変換の合成。
        """
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
