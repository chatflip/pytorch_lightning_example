import os

import pytorch_lightning as L
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from Food101Dataset import Food101Dataset
from Food101Downloader import Food101Downloader


class ClassificationDataModule(L.LightningDataModule):
    """分類タスク用のPyTorch Lightning DataModule

    このモジュールは分類データセットのデータ準備、読み込み、変換を処理する。
    Food101データセット専用に設計されている。
    """

    def __init__(
        self,
        args: DictConfig,
        model_cfg: dict[str, object],
    ) -> None:
        """ClassificationDataModuleを初期化する

        Args:
            args: dataset_root、batch_size、debugなどを含む設定引数。
            model_cfg: input_size、crop_pct、mean、stdを含むモデル設定辞書。
        """
        super().__init__()
        self.args = args
        model_input_size: tuple[int, ...] = tuple(model_cfg["input_size"])  # type: ignore[assignment]
        model_crop_cnt: float = float(model_cfg["crop_pct"])  # type: ignore[assignment]
        self.input_height = model_input_size[1]
        self.input_width = model_input_size[2]
        self.crop_height = int(round(self.input_height / model_crop_cnt))
        self.crop_width = int(round(self.input_width / model_crop_cnt))
        self.model_mean = model_cfg["mean"]
        self.model_std = model_cfg["std"]

    def prepare_data(self) -> None:
        """データセットをダウンロードして準備する

        このメソッドは単一のGPU/プロセスからのみ呼び出される。
        """
        Food101Downloader().download()

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
            num_workers=os.cpu_count() or 0,
            pin_memory=True,
            drop_last=train,
        )

    @property
    def train_transforms(self) -> transforms.Compose:
        """訓練用データ変換を取得する

        Returns:
            transforms.Compose: TrivialAugmentWide、Resize、RandomCrop、
                RandomHorizontalFlip、ToTensor、Normalizeを含む訓練用変換の合成。
        """
        return transforms.Compose(
            [
                transforms.TrivialAugmentWide(),
                transforms.Resize(
                    (self.crop_height, self.crop_width), InterpolationMode.BILINEAR
                ),  # リサイズ
                transforms.RandomCrop(
                    (self.input_height, self.input_width)
                ),  # クロップ
                transforms.RandomHorizontalFlip(p=0.5),  # 左右反転
                transforms.ToTensor(),  # テンソル化
                transforms.Normalize(
                    mean=self.model_mean,
                    std=self.model_std,
                ),
            ]
        )

    @property
    def val_transforms(self) -> transforms.Compose:
        """検証/テスト用データ変換を取得する

        Returns:
            transforms.Compose: Resize、CenterCrop、ToTensor、Normalizeを含む
                検証用変換の合成。
        """
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
