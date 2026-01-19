from typing import Literal, Union, get_args, get_origin

import pytest
from pytorch_lightning.accelerators import AcceleratorRegistry
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.strategies import StrategyRegistry

from config.schema import (
    CheckpointConfig,
    TrainerConfig,
)


def extract_literal_values(annotation: type) -> set:
    """型アノテーションから許可される値を抽出する.

    例: `bool | Literal["warn"]` → {True, False, "warn"}
    """
    values: set = set()

    origin = get_origin(annotation)

    if origin is Union:
        for arg in get_args(annotation):
            values.update(extract_literal_values(arg))
    elif origin is Literal:
        values.update(get_args(annotation))
    elif annotation is bool:
        values.update({True, False})
    else:
        pass

    return values


class TestAcceleratorValues:
    """acceleratorの有効な値をテスト."""

    SCHEMA_VALUES = extract_literal_values(
        TrainerConfig.model_fields["accelerator"].annotation
    )

    SPECIAL_VALUES = {"auto", "gpu"}  # "gpu" は "cuda" のエイリアス

    def test_schema_values_are_valid(self) -> None:
        """スキーマで定義した値がすべてレジストリに存在するか確認."""
        registered = set(AcceleratorRegistry.keys())
        valid_values = registered | self.SPECIAL_VALUES

        for value in self.SCHEMA_VALUES:
            assert value in valid_values, (
                f"accelerator='{value}' is not valid. "
                f"Valid values are: {valid_values}"
            )

    def test_no_missing_accelerators(self) -> None:
        """レジストリにあってスキーマにない値がないか確認."""
        registered = set(AcceleratorRegistry.keys())
        expected = registered | self.SPECIAL_VALUES

        missing = expected - self.SCHEMA_VALUES
        assert not missing, (
            f"Schema is missing accelerator values: {missing}. "
            f"Consider adding them to the schema."
        )


class TestStrategyValues:
    """strategyの有効な値をテスト."""

    SCHEMA_VALUES = extract_literal_values(
        TrainerConfig.model_fields["strategy"].annotation
    )

    SPECIAL_VALUES = {"auto"}

    BASE_STRATEGIES = {
        "ddp",
        "ddp_spawn",
        "ddp_fork",
        "ddp_notebook",
        "fsdp",
        "deepspeed",
        "single_device",
    }

    def test_schema_values_are_valid(self) -> None:
        """スキーマで定義した値がすべてレジストリに存在するか確認."""
        registered = set(StrategyRegistry.keys())
        valid_values = registered | self.SPECIAL_VALUES

        for value in self.SCHEMA_VALUES:
            assert value in valid_values, (
                f"strategy='{value}' is not valid. "
                f"Valid values are: {valid_values}"
            )

    def test_base_strategies_covered(self) -> None:
        """基本的な戦略がスキーマでカバーされているか確認."""
        essential = {"ddp", "ddp_spawn", "fsdp", "deepspeed"}
        missing = essential - self.SCHEMA_VALUES
        assert not missing, f"Schema is missing essential strategies: {missing}"


class TestPrecisionValues:
    """precisionの有効な値をテスト.

    precisionはレジストリがないため、ドキュメントベースで確認。
    """

    SCHEMA_VALUES = extract_literal_values(
        TrainerConfig.model_fields["precision"].annotation
    )

    VALID_PRECISION_VALUES = {
        "64",
        "32",
        "64-true",
        "32-true",
        "16",
        "bf16",
        "16-true",
        "bf16-true",
        "16-mixed",
        "bf16-mixed",
    }

    def test_schema_values_are_valid(self) -> None:
        """スキーマで定義した値がすべて有効か確認."""
        invalid = self.SCHEMA_VALUES - self.VALID_PRECISION_VALUES
        assert not invalid, f"Schema contains invalid precision values: {invalid}"

    def test_all_valid_values_covered(self) -> None:
        """有効な値がすべてスキーマでカバーされているか確認."""
        missing = self.VALID_PRECISION_VALUES - self.SCHEMA_VALUES
        assert not missing, f"Schema is missing precision values: {missing}"


class TestCheckpointModeValues:
    """ModelCheckpointのmodeの有効な値をテスト."""

    SCHEMA_VALUES = list(
        extract_literal_values(CheckpointConfig.model_fields["mode"].annotation)
    )

    @pytest.mark.parametrize("mode", SCHEMA_VALUES)
    def test_mode_values_accepted_by_checkpoint(self, mode: str) -> None:
        """スキーマで定義したmode値がModelCheckpointに受け入れられるかテスト."""
        checkpoint = ModelCheckpoint(
            monitor="val_loss",
            mode=mode,
        )
        assert checkpoint.mode == mode


class TestProgressBarRefreshRate:
    """TQDMProgressBarのrefresh_rateをテスト."""

    @pytest.mark.parametrize("refresh_rate", [1, 10, 100])
    def test_refresh_rate_values_accepted(self, refresh_rate: int) -> None:
        """スキーマで定義したrefresh_rate値がTQDMProgressBarに受け入れられるかテスト."""
        progress_bar = TQDMProgressBar(refresh_rate=refresh_rate)
        assert progress_bar.refresh_rate == refresh_rate


class TestDeterministicValues:
    """deterministicの有効な値をテスト."""

    VALID_VALUES = {True, False, "warn"}

    SCHEMA_VALUES = extract_literal_values(
        TrainerConfig.model_fields["deterministic"].annotation
    )

    def test_schema_values_are_valid(self) -> None:
        """スキーマで定義した値がすべて有効か確認."""
        invalid = self.SCHEMA_VALUES - self.VALID_VALUES
        assert not invalid, f"Schema contains invalid deterministic values: {invalid}"

    def test_all_valid_values_covered(self) -> None:
        """有効な値がすべてスキーマでカバーされているか確認."""
        missing = self.VALID_VALUES - self.SCHEMA_VALUES
        assert not missing, f"Schema is missing deterministic values: {missing}"


class TestValCheckIntervalCombination:
    """val_check_intervalとcheck_val_every_n_epochの組み合わせをテスト."""

    def test_check_val_every_n_epoch_none_with_float_interval_succeeds(self) -> None:
        """check_val_every_n_epoch=Noneでval_check_intervalがfloatの場合、スキーマでは成功する."""
        # スキーマではエラーにならない（build_trainerで解決される）
        config = TrainerConfig(
            check_val_every_n_epoch=None,
            val_check_interval=0.5,
        )
        assert config.check_val_every_n_epoch is None
        assert config.val_check_interval == 0.5

    def test_check_val_every_n_epoch_none_with_int_interval_succeeds(self) -> None:
        """check_val_every_n_epoch=Noneでval_check_intervalが整数の場合、成功する."""
        config = TrainerConfig(
            check_val_every_n_epoch=None,
            val_check_interval=100,
        )
        assert config.check_val_every_n_epoch is None
        assert config.val_check_interval == 100

    def test_check_val_every_n_epoch_set_with_float_interval_succeeds(self) -> None:
        """check_val_every_n_epochが設定されていてval_check_intervalがfloatの場合、成功する."""
        config = TrainerConfig(
            check_val_every_n_epoch=2,
            val_check_interval=0.5,
        )
        assert config.check_val_every_n_epoch == 2
        assert config.val_check_interval == 0.5

    def test_check_val_every_n_epoch_set_with_int_interval_succeeds(self) -> None:
        """check_val_every_n_epochが設定されていてval_check_intervalが整数の場合、成功する."""
        config = TrainerConfig(
            check_val_every_n_epoch=2,
            val_check_interval=100,
        )
        assert config.check_val_every_n_epoch == 2
        assert config.val_check_interval == 100

    def test_default_values_succeed(self) -> None:
        """デフォルト値の場合、成功する."""
        config = TrainerConfig()
        assert config.check_val_every_n_epoch is None
        assert config.val_check_interval == 1.0
        # デフォルトではcheck_val_every_n_epoch=Noneでval_check_interval=1.0（float）なので
        # val_check_intervalが使用される（毎エポックの終わりにバリデーション）


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
