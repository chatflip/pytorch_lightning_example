import os
from pathlib import Path
from typing import Any

import pytorch_lightning as L
from torch.utils.data import DataLoader

from builders.augmentation import build_transforms
from data.dataset import ImageFolderDataset


class ClassificationDataModule(L.LightningDataModule):
    """分類タスク用のPyTorch Lightning DataModule.

    YAML設定からデータローダーを構築する。
    ImageFolder形式のデータセット構造を想定:
        dataset_root/
        ├── train/
        │   ├── class_a/
        │   └── ...
        └── val/
            ├── class_a/
            └── ...
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """ClassificationDataModuleを初期化する.

        Args:
            config: 設定辞書。以下のキーを含む:
                - data.dataset_root: データセットのルートパス
                - data.batch_size: バッチサイズ
                - data.num_workers: ワーカー数
                - data.pin_memory: ピンメモリの使用
                - augmentation.train: 訓練用オーギュメンテーション設定
                - augmentation.val: 検証用オーギュメンテーション設定
        """
        super().__init__()
        self.config = config
        self.data_config = config.get("data", {})
        self.aug_config = config.get("augmentation", {})

        self.dataset_root = Path(self.data_config.get("dataset_root", "./datasets"))
        self.batch_size = self.data_config.get("batch_size", 32)
        self.num_workers = self.data_config.get("num_workers", os.cpu_count() or 4)
        self.pin_memory = self.data_config.get("pin_memory", True)

        self.train_transform = None
        self.val_transform = None
        if "train" in self.aug_config:
            self.train_transform = build_transforms(self.aug_config["train"])
        if "val" in self.aug_config:
            self.val_transform = build_transforms(self.aug_config["val"])

        self.train_dataset: ImageFolderDataset | None = None
        self.val_dataset: ImageFolderDataset | None = None
        self.test_dataset: ImageFolderDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """データセットをセットアップする.

        Args:
            stage: "fit", "validate", "test", または "predict"
        """
        if stage == "fit" or stage is None:
            train_root = self.dataset_root / "train"
            val_root = self.dataset_root / "val"

            self.train_dataset = ImageFolderDataset(
                root=train_root,
                transform=self.train_transform,
            )
            self.val_dataset = ImageFolderDataset(
                root=val_root,
                transform=self.val_transform,
            )

        if stage == "test" or stage is None:
            test_root = self.dataset_root / "test"
            if not test_root.exists():
                test_root = self.dataset_root / "val"

            self.test_dataset = ImageFolderDataset(
                root=test_root,
                transform=self.val_transform,
            )

    def train_dataloader(self) -> DataLoader:
        """訓練用データローダーを返す."""
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is not initialized. Call setup() first.")

        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """検証用データローダーを返す."""
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is not initialized. Call setup() first.")

        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        """テスト用データローダーを返す."""
        if self.test_dataset is None:
            raise RuntimeError("test_dataset is not initialized. Call setup() first.")

        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    @property
    def num_classes(self) -> int:
        """クラス数を返す."""
        num_classes = self.data_config.get("num_classes")
        if num_classes is not None:
            return num_classes

        if self.train_dataset is not None:
            return self.train_dataset.num_classes
        if self.val_dataset is not None:
            return self.val_dataset.num_classes

        raise RuntimeError(
            "Cannot determine num_classes. "
            "Call setup() first or set data.num_classes in config."
        )

    @property
    def classes(self) -> list[str]:
        """クラス名のリストを返す."""
        if self.train_dataset is not None:
            return self.train_dataset.classes
        if self.val_dataset is not None:
            return self.val_dataset.classes
        raise RuntimeError("Cannot get classes. Call setup() first.")
