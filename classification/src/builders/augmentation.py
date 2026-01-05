from typing import Any

import albumentations as A
from albumentations.core.composition import BaseCompose
from albumentations.pytorch import ToTensorV2

from config import (
    ConfigValidationError,
    TransformOpConfig,
    validate_transform_config,
)


def build_transforms(
    config: dict[str, Any],
    model_config: dict[str, Any] | None = None,
) -> A.Compose:
    """YAML設定からalbumentationsパイプラインを構築する.

    Args:
        config: ops配列を含む辞書
            例: {"ops": [{"type": "Resize", "height": 256, "width": 256}, ...]}
        model_config: モデル設定辞書。input_sizeを含む場合、
            augmentationのサイズを自動調整する。

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

    input_size: int | None = None

    if model_config:
        input_size = model_config["input_size"]

    for i, op_config in enumerate(transform_cfg.ops):
        try:
            transform = _build_single_transform(
                op_config,
                input_size=input_size,
            )
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


def _raise_size_error(op_type: str, op_args: dict[str, Any]) -> None:
    """サイズ未指定エラーを発生させる."""
    error_msg = "height/widthが未指定です。model.input_sizeを設定してください"
    raise ConfigValidationError(
        section="augmentation",
        errors=[
            {
                "loc": ["ops", op_type],
                "msg": error_msg,
                "input": op_args,
            }
        ],
    )


def _apply_size_defaults(
    op_type: str,
    op_args: dict[str, Any],
    input_size: int | None,
) -> dict[str, Any]:
    """サイズ関連のデフォルト値を適用する."""
    if "height" in op_args and "width" in op_args:
        return op_args

    if op_type in ("RandomResizedCrop", "CenterCrop", "Resize"):
        if input_size is None:
            _raise_size_error(op_type, op_args)
        op_args["height"] = input_size
        op_args["width"] = input_size

    return op_args


def _build_single_transform(
    op_config: TransformOpConfig,
    input_size: int | None = None,
) -> A.BasicTransform:
    """単一のトランスフォームを構築する.

    Args:
        op_config: バリデーション済みのTransformOpConfig
        input_size: モデルの入力サイズ（height/widthが未指定の場合に使用）

    Returns:
        albumentationsトランスフォームオブジェクト

    Raises:
        ConfigValidationError: トランスフォームの構築に失敗した場合
    """
    op_type = op_config.type
    op_args = {k: v for k, v in op_config.model_dump().items() if k != "type"}

    op_args = _apply_size_defaults(op_type, op_args, input_size)

    if op_type == "RandomResizedCrop" and "height" in op_args and "width" in op_args:
        op_args["size"] = (op_args.pop("height"), op_args.pop("width"))

    return _instantiate_transform(op_type, op_args)


def _instantiate_transform(op_type: str, op_args: dict[str, Any]) -> A.BasicTransform:
    """トランスフォームクラスをインスタンス化する."""
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
