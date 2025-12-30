from typing import Any

import albumentations as A
from albumentations.core.composition import BaseCompose
from albumentations.pytorch import ToTensorV2

from config import (
    ConfigValidationError,
    TransformOpConfig,
    validate_transform_config,
)


def build_transforms(config: dict[str, Any]) -> A.Compose:
    """YAML設定からalbumentationsパイプラインを構築する.

    Args:
        config: ops配列を含む辞書
            例: {"ops": [{"type": "Resize", "height": 256, "width": 256}, ...]}

    Returns:
        albumentations.Compose オブジェクト

    Raises:
        ConfigValidationError: 設定のバリデーションに失敗した場合

    Example:
        >>> config = {"ops": [{"type": "Resize", "height": 256, "width": 256}]}
        >>> transform = build_transforms(config)
    """
    transform_cfg = validate_transform_config(config)
    transforms: list[A.BasicTransform | BaseCompose] = []

    for i, op_config in enumerate(transform_cfg.ops):
        try:
            transform = _build_single_transform(op_config)
            transforms.append(transform)
        except ConfigValidationError:
            raise
        except Exception as e:
            raise ConfigValidationError(
                section="augmentation",
                errors=[
                    {
                        "loc": ["ops", i, op_config.type],
                        "msg": str(e),
                        "input": op_config.model_dump(),
                    }
                ],
            ) from e

    try:
        return A.Compose(transforms)
    except Exception as e:
        raise ConfigValidationError(
            section="augmentation",
            errors=[
                {
                    "loc": ["Compose"],
                    "msg": str(e),
                    "input": "N/A",
                }
            ],
        ) from e


def _build_single_transform(op_config: TransformOpConfig) -> A.BasicTransform:
    """単一のトランスフォームを構築する.

    Args:
        op_config: バリデーション済みのTransformOpConfig

    Returns:
        albumentationsトランスフォームオブジェクト

    Raises:
        ConfigValidationError: トランスフォームの構築に失敗した場合
    """
    op_type = op_config.type
    op_args = {k: v for k, v in op_config.model_dump().items() if k != "type"}

    try:
        if op_type == "ToTensorV2":
            return ToTensorV2(**op_args)

        transform_class = getattr(A, op_type)
        return transform_class(**op_args)
    except AttributeError as e:
        raise ConfigValidationError(
            section="augmentation",
            errors=[
                {
                    "loc": ["ops", op_type],
                    "msg": f"不明なトランスフォームタイプ: {op_type}",
                    "input": op_type,
                }
            ],
        ) from e
    except Exception as e:
        raise ConfigValidationError(
            section="augmentation",
            errors=[
                {
                    "loc": ["ops", op_type],
                    "msg": str(e),
                    "input": op_args,
                }
            ],
        ) from e


SUPPORTED_TRANSFORMS = [
    "Resize",
    "RandomResizedCrop",
    "CenterCrop",
    "RandomCrop",
    "Crop",
    "LongestMaxSize",
    "SmallestMaxSize",
    "PadIfNeeded",
    "HorizontalFlip",
    "VerticalFlip",
    "Rotate",
    "RandomRotate90",
    "Affine",
    "ShiftScaleRotate",
    "ColorJitter",
    "RandomBrightnessContrast",
    "HueSaturationValue",
    "RGBShift",
    "ChannelShuffle",
    "CLAHE",
    "Equalize",
    "ToGray",
    "ToSepia",
    "GaussianBlur",
    "MotionBlur",
    "MedianBlur",
    "Blur",
    "GaussNoise",
    "ISONoise",
    "ImageCompression",
    "CoarseDropout",
    "GridDropout",
    "PixelDropout",
    "GridDistortion",
    "ElasticTransform",
    "OpticalDistortion",
    "Perspective",
    "Normalize",
    "ToTensorV2",
]
