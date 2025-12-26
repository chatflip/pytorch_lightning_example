"""オーギュメンテーションビルダー.

YAML設定からalbumentationsパイプラインを構築する。
"""

from typing import Any

import albumentations as A
from albumentations.core.composition import BaseCompose
from albumentations.pytorch import ToTensorV2


def build_transforms(config: dict[str, Any]) -> A.Compose:
    """YAML設定からalbumentationsパイプラインを構築する.

    Args:
        config: ops配列を含む辞書
            例: {"ops": [{"type": "Resize", "height": 256, "width": 256}, ...]}

    Returns:
        albumentations.Compose オブジェクト

    Example:
        >>> config = {"ops": [{"type": "Resize", "height": 256, "width": 256}]}
        >>> transform = build_transforms(config)
    """
    ops = config.get("ops", [])
    transforms: list[A.BasicTransform | BaseCompose] = []

    for op_config in ops:
        op_config = op_config.copy()  # 元の辞書を変更しないようにコピー
        op_type = op_config.pop("type")
        transform = _build_single_transform(op_type, op_config)
        transforms.append(transform)

    return A.Compose(transforms)


def _build_single_transform(op_type: str, op_args: dict[str, Any]) -> A.BasicTransform:
    """単一のトランスフォームを構築する.

    Args:
        op_type: トランスフォームのクラス名
        op_args: トランスフォームの引数

    Returns:
        albumentationsトランスフォームオブジェクト

    Raises:
        ValueError: 未知のトランスフォームタイプの場合
    """
    # ToTensorV2は特別扱い
    if op_type == "ToTensorV2":
        return ToTensorV2(**op_args)

    # albumentationsモジュールからクラスを取得
    if hasattr(A, op_type):
        transform_class = getattr(A, op_type)
        return transform_class(**op_args)
    else:
        raise ValueError(f"Unknown transform type: {op_type}")


# サポートするトランスフォームの一覧（ドキュメント用）
SUPPORTED_TRANSFORMS = [
    # リサイズ系
    "Resize",
    "RandomResizedCrop",
    "CenterCrop",
    "RandomCrop",
    "Crop",
    "LongestMaxSize",
    "SmallestMaxSize",
    "PadIfNeeded",
    # 反転・回転
    "HorizontalFlip",
    "VerticalFlip",
    "Rotate",
    "RandomRotate90",
    "Affine",
    "ShiftScaleRotate",
    # 色変換
    "ColorJitter",
    "RandomBrightnessContrast",
    "HueSaturationValue",
    "RGBShift",
    "ChannelShuffle",
    "CLAHE",
    "Equalize",
    "ToGray",
    "ToSepia",
    # ノイズ・ブラー
    "GaussianBlur",
    "MotionBlur",
    "MedianBlur",
    "Blur",
    "GaussNoise",
    "ISONoise",
    "ImageCompression",
    # ドロップアウト
    "CoarseDropout",
    "GridDropout",
    "PixelDropout",
    # 歪み
    "GridDistortion",
    "ElasticTransform",
    "OpticalDistortion",
    "Perspective",
    # 正規化
    "Normalize",
    "ToTensorV2",
]
