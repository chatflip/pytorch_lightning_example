import numpy as np
import pytest

import albumentations as A

from builders.augmentation import (
    _build_single_transform,
    build_transforms,
)
from config import ConfigValidationError, TransformOpConfig


class TestRandomResizedCropSizeConversion:
    """RandomResizedCropのheight/width→size変換テスト."""

    def test_height_width_converted_to_size(self) -> None:
        """height/widthがsizeタプルに正しく変換されることを確認."""
        config = TransformOpConfig(
            type="RandomResizedCrop",
            height=224,
            width=224,
            scale=[0.08, 1.0],
            ratio=[0.75, 1.333],
        )
        transform = _build_single_transform(config)

        assert isinstance(transform, A.RandomResizedCrop)
        assert transform.size == (224, 224)

    def test_asymmetric_size(self) -> None:
        """非正方形サイズも正しく変換されることを確認."""
        config = TransformOpConfig(
            type="RandomResizedCrop",
            height=224,
            width=192,
        )
        transform = _build_single_transform(config)

        assert transform.size == (224, 192)


class TestCropTransformsKeepHeightWidth:
    """CenterCrop/RandomCropがheight/widthを保持することのテスト."""

    def test_center_crop_uses_height_width(self) -> None:
        """CenterCropはheight/widthパラメータを使用することを確認."""
        config = TransformOpConfig(
            type="CenterCrop",
            height=224,
            width=224,
        )
        transform = _build_single_transform(config)

        assert isinstance(transform, A.CenterCrop)
        assert transform.height == 224
        assert transform.width == 224

    def test_random_crop_uses_height_width(self) -> None:
        """RandomCropはheight/widthパラメータを使用することを確認."""
        config = TransformOpConfig(
            type="RandomCrop",
            height=224,
            width=224,
        )
        transform = _build_single_transform(config)

        assert isinstance(transform, A.RandomCrop)
        assert transform.height == 224
        assert transform.width == 224


class TestBuildTransforms:
    """build_transforms関数のテスト."""

    def test_build_basic_pipeline(self) -> None:
        """基本的なパイプラインが構築できることを確認."""
        config = {
            "ops": [
                {"type": "Resize", "height": 256, "width": 256},
                {"type": "HorizontalFlip", "p": 0.5},
                {"type": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                {"type": "ToTensorV2"},
            ]
        }
        transform = build_transforms(config)

        assert isinstance(transform, A.Compose)
        assert len(transform.transforms) == 4

    def test_build_with_random_resized_crop(self) -> None:
        """RandomResizedCropを含むパイプラインが構築できることを確認."""
        config = {
            "ops": [
                {"type": "RandomResizedCrop", "height": 224, "width": 224, "scale": [0.08, 1.0]},
                {"type": "HorizontalFlip", "p": 0.5},
                {"type": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                {"type": "ToTensorV2"},
            ]
        }
        transform = build_transforms(config)

        assert isinstance(transform, A.Compose)
        assert isinstance(transform.transforms[0], A.RandomResizedCrop)

    def test_build_with_center_crop(self) -> None:
        """CenterCropを含むパイプラインが構築できることを確認."""
        config = {
            "ops": [
                {"type": "Resize", "height": 256, "width": 256},
                {"type": "CenterCrop", "height": 224, "width": 224},
                {"type": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                {"type": "ToTensorV2"},
            ]
        }
        transform = build_transforms(config)

        assert isinstance(transform, A.Compose)
        assert isinstance(transform.transforms[1], A.CenterCrop)

    def test_pipeline_execution(self) -> None:
        """構築したパイプラインが実際に画像に適用できることを確認."""
        config = {
            "ops": [
                {"type": "Resize", "height": 256, "width": 256},
                {"type": "CenterCrop", "height": 224, "width": 224},
                {"type": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                {"type": "ToTensorV2"},
            ]
        }
        transform = build_transforms(config)

        dummy_image = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
        result = transform(image=dummy_image)

        assert "image" in result
        assert result["image"].shape == (3, 224, 224)


class TestColorJitter:
    """ColorJitterのテスト."""

    def test_color_jitter_build(self) -> None:
        """ColorJitterが正しく構築できることを確認."""
        config = TransformOpConfig(
            type="ColorJitter",
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1,
            p=0.5,
        )
        transform = _build_single_transform(config)

        assert isinstance(transform, A.ColorJitter)
        assert transform.p == 0.5


class TestGaussianBlur:
    """GaussianBlurのテスト."""

    def test_gaussian_blur_build(self) -> None:
        """GaussianBlurが正しく構築できることを確認."""
        config = TransformOpConfig(
            type="GaussianBlur",
            blur_limit=[3, 7],
            p=0.1,
        )
        transform = _build_single_transform(config)

        assert isinstance(transform, A.GaussianBlur)
        assert transform.p == 0.1


class TestErrorHandling:
    """エラーハンドリングのテスト."""

    def test_unknown_transform_type_raises_error(self) -> None:
        """不明なtransformタイプでConfigValidationErrorが発生することを確認."""
        config = {"ops": [{"type": "UnknownTransform"}]}

        with pytest.raises(ConfigValidationError) as exc_info:
            build_transforms(config)

        assert "不明なトランスフォームタイプ" in str(exc_info.value)

    def test_invalid_parameter_raises_error(self) -> None:
        """不正なパラメータでエラーが発生することを確認."""
        config = {"ops": [{"type": "Resize", "height": -1, "width": 256}]}

        with pytest.raises(ConfigValidationError):
            build_transforms(config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

