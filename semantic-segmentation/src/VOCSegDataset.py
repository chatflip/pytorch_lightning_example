# -*- coding: utf-8 -*-
import os
from typing import Callable

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class VOCSegDataset(Dataset):
    """VOC2012セグメンテーションデータセット用のDatasetクラス

    このデータセットはVOC2012データセット構造から画像とセグメンテーションマスクを
    読み込み、マスクをワンホットエンコード形式に変換する。
    """

    def __init__(
        self,
        root: str,
        phase: str,
        num_classes: int,
        transform: Callable[
            [np.ndarray, np.ndarray], dict[str, torch.Tensor | np.ndarray]
        ]
        | None = None,
    ) -> None:
        """VOCSegDatasetを初期化する

        Args:
            root: VOC2012データセット構造を含むルートディレクトリパス。
            phase: データセットのフェーズ。'train'または'val'。
            num_classes: セグメンテーションクラス数。
            transform: 画像とマスクに適用するオプションの変換。
        """
        self.transform = transform  # 画像変形用
        self.image_paths: list[str] = []  # 画像のパス格納用
        self.mask_paths: list[str] = []  # 画像のラベル格納用
        self.num_classes = num_classes
        image_list_path = os.path.join(
            root, "ImageSets", "Segmentation", f"{phase}.txt"
        )
        image_lists = pd.read_table(image_list_path, header=None)

        for _, image_list in image_lists.iterrows():
            image_path = f"{root}/JPEGImages/{image_list.values[0]}.jpg"
            mask_path = f"{root}/SegmentationRaw/{image_list.values[0]}.png"
            self.image_paths.append(image_path)
            self.mask_paths.append(mask_path)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """データセットからサンプルを取得する

        Args:
            index: 取得するサンプルのインデックス。

        Returns:
            変換された画像とワンホットエンコードされたマスクを含むタプル。
        """
        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[index], 0)
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)  # 画像変形適用
            image, mask = augmented["image"], augmented["mask"]
        else:
            # If no transform, convert to tensor manually
            from albumentations.pytorch import ToTensorV2
            transform = ToTensorV2()
            augmented = transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
        height, width = mask.shape
        binary_mask = torch.zeros(self.num_classes, height, width)
        for i in range(self.num_classes):
            one_hot_mask = torch.where(mask == i, 1, 0)
            binary_mask[i, :, :] = one_hot_mask
        return image, binary_mask  # 画像とラベルを返す

    def __len__(self) -> int:
        """データセットのサイズを返す

        Returns:
            データセット内のサンプル数。
        """
        return len(self.image_paths)
