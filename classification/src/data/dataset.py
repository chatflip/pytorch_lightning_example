"""汎用ImageFolder形式データセット.

albumentationsトランスフォームに対応した画像分類データセット。
"""

from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# サポートする画像拡張子
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}


class ImageFolderDataset(Dataset[tuple[torch.Tensor, int]]):
    """ImageFolder形式のデータセット.

    ディレクトリ構造:
        root/
        ├── class_a/
        │   ├── image_001.jpg
        │   └── ...
        ├── class_b/
        │   └── ...
        └── ...

    Attributes:
        root: データセットのルートディレクトリ
        transform: albumentationsトランスフォーム
        classes: クラス名のリスト
        class_to_idx: クラス名からインデックスへのマッピング
        samples: (画像パス, ラベル) のリスト
    """

    def __init__(
        self,
        root: str | Path,
        transform: A.Compose | None = None,
    ) -> None:
        """ImageFolderDatasetを初期化する.

        Args:
            root: データセットのルートディレクトリ
            transform: albumentationsトランスフォーム（オプション）
        """
        self.root = Path(root)
        self.transform = transform

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        # クラス名を取得（ディレクトリ名がクラス名）
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        if len(self.classes) == 0:
            raise ValueError(f"No class directories found in {self.root}")

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # サンプルを収集
        self.samples: list[tuple[Path, int]] = []
        for class_name in self.classes:
            class_dir = self.root / class_name
            class_idx = self.class_to_idx[class_name]

            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((img_path, class_idx))

        if len(self.samples) == 0:
            raise ValueError(f"No images found in {self.root}")

    def __len__(self) -> int:
        """データセットのサイズを返す."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """指定されたインデックスのサンプルを取得する.

        Args:
            index: サンプルのインデックス

        Returns:
            (画像テンソル, ラベル) のタプル
        """
        img_path, label = self.samples[index]

        # 画像を読み込み（RGB形式）
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)

        # albumentationsトランスフォームを適用
        if self.transform is not None:
            transformed: dict[str, Any] = self.transform(image=image_np)
            image_tensor: torch.Tensor = transformed["image"]
        else:
            # transformがない場合は単純にテンソル化
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

        return image_tensor, label

    def get_class_name(self, label: int) -> str:
        """ラベルからクラス名を取得する.

        Args:
            label: クラスのインデックス

        Returns:
            クラス名
        """
        return self.classes[label]

    @property
    def num_classes(self) -> int:
        """クラス数を返す."""
        return len(self.classes)

    def __repr__(self) -> str:
        """データセットの文字列表現."""
        return (
            f"{self.__class__.__name__}(\n"
            f"  root={self.root},\n"
            f"  num_samples={len(self.samples)},\n"
            f"  num_classes={self.num_classes},\n"
            f"  classes={self.classes[:5]}{'...' if len(self.classes) > 5 else ''}\n"
            f")"
        )
