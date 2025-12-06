import json
import os
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset


class Food101Dataset(Dataset):
    """Food101分類データセット用のDatasetクラス

    このデータセットはFood101データセット構造から画像とラベルを読み込む。
    """

    def __init__(
        self,
        root: str,
        phase: str,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        """Food101Datasetを初期化する

        Args:
            root: 'images'と'meta'サブディレクトリを含むルートディレクトリパス。
            phase: データセットのフェーズ。'train'または'val'/'test'。
            transform: 画像に適用するオプションの変換。
        """
        self.transform = transform
        self.image_paths: list[str] = []
        self.image_labels: list[int] = []
        image_root = os.path.join(root, "images")
        metadata_root = os.path.join(root, "meta")
        if phase == "train":
            image_dict_path = os.path.join(metadata_root, "train.json")
        else:
            image_dict_path = os.path.join(metadata_root, "test.json")
        class_name_path = os.path.join(metadata_root, "classes.txt")
        with open(class_name_path, newline=None, mode="r") as f:
            class_names = [s.strip() for s in f.readlines()]

        with open(image_dict_path) as f:
            image_dict = json.load(f)
        for i, class_name in enumerate(class_names):
            filenames = image_dict[class_name]
            image_paths = [os.path.join(image_root, f"{f}.jpg") for f in filenames]
            self.image_labels.extend([i] * len(filenames))
            self.image_paths.extend(image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """データセットからサンプルを取得する

        Args:
            index: 取得するサンプルのインデックス。

        Returns:
            変換された画像とそのラベルを含むタプル。
        """
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
            assert isinstance(image, torch.Tensor), "Transform must return torch.Tensor"
        else:
            # transformがNoneの場合はToTensorを適用
            from torchvision import transforms

            image = transforms.ToTensor()(image)
        return image, self.image_labels[index]

    def __len__(self) -> int:
        """データセットのサイズを返す

        Returns:
            データセット内のサンプル数。
        """
        return len(self.image_paths)
