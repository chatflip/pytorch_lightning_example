from typing import Any, Literal

import albumentations as A
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


class CheckpointConfig(BaseModel):
    """チェックポイント設定のバリデーションスキーマ."""

    monitor: str = Field(
        default="val_loss",
        description="監視するメトリクス名",
    )
    mode: Literal["min", "max"] = Field(
        default="min",
        description="監視モード（'min': 最小値が最良, 'max': 最大値が最良）",
    )
    save_top_k: int = Field(
        default=1,
        ge=-1,
        description="保存するチェックポイント数（-1: 全て保存, 0: 保存しない）",
    )
    save_last: bool = Field(
        default=True,
        description="最後のエポックのチェックポイントを保存するか",
    )

    @field_validator("monitor")
    @classmethod
    def validate_monitor(cls, v: str) -> str:
        """monitorが空でないことを検証."""
        if not v or not v.strip():
            raise ValueError("monitor は空にできません")
        return v


class ProgressBarConfig(BaseModel):
    """プログレスバー設定のバリデーションスキーマ."""

    refresh_rate: int = Field(
        default=10,
        ge=1,
        description="プログレスバーの更新頻度（バッチ数）",
    )


class TrainerConfig(BaseModel):
    """トレーナー設定のバリデーションスキーマ."""

    max_epochs: int = Field(
        default=100,
        ge=1,
        description="最大エポック数",
    )
    accelerator: Literal["cpu", "gpu", "cuda", "tpu", "mps", "auto"] = Field(
        default="gpu",
        description="使用するアクセラレータ（'gpu'と'cuda'は同義）",
    )
    devices: int | Literal["auto"] = Field(
        default=1,
        description="使用するデバイス数（'auto': 自動選択）",
    )
    precision: Literal[
        "64",
        "32",
        "16",
        "bf16",
        "64-true",
        "32-true",
        "16-true",
        "bf16-true",
        "16-mixed",
        "bf16-mixed",
    ] = Field(
        default="16-mixed",
        description="計算精度",
    )
    strategy: Literal[
        "auto",
        "ddp",
        "ddp_spawn",
        "ddp_notebook",
        "fsdp",
        "deepspeed",
    ] = Field(
        default="auto",
        description="分散学習戦略",
    )
    deterministic: bool | Literal["warn"] = Field(
        default=True,
        description="決定論的な計算を行うか（'warn': 警告のみ）",
    )
    log_every_n_steps: int = Field(
        default=10,
        ge=1,
        description="ログを出力するステップ間隔",
    )
    val_check_interval: float | int = Field(
        default=1.0,
        description="検証を行う間隔（0.0-1.0: エポックの割合, >=1: バッチ数）",
    )

    @field_validator("devices")
    @classmethod
    def validate_devices(cls, v: int | str) -> int | str:
        """devicesの値を検証."""
        if isinstance(v, int) and v < 1:
            raise ValueError("devicesは1以上の整数または'auto'である必要があります")
        return v

    @field_validator("val_check_interval")
    @classmethod
    def validate_val_check_interval(cls, v: float | int) -> float | int:
        """val_check_intervalの値を検証."""
        if isinstance(v, float) and (v <= 0 or v > 1.0):
            if v < 1.0:
                raise ValueError(
                    "val_check_interval は 0.0 より大きく 1.0 以下の小数、"
                    "または 1 以上の整数である必要があります"
                )
        if isinstance(v, int) and v < 1:
            raise ValueError(
                "val_check_interval は 0.0 より大きく 1.0 以下の小数、"
                "または 1 以上の整数である必要があります"
            )
        return v


class LoggerConfig(BaseModel):
    """ロガー設定のバリデーションスキーマ."""

    type: Literal["mlflow", "tensorboard"] = Field(
        default="mlflow",
        description="ロガータイプ",
    )
    experiment_name: str = Field(
        default="classification",
        description="実験名（MLFlowの場合）",
    )
    tracking_uri: str | None = Field(
        default=None,
        description="MLFlowトラッキングURI",
    )
    log_model: bool = Field(
        default=True,
        description="モデルをログするか（MLFlowの場合）",
    )
    name: str = Field(
        default="tensorboard",
        description="ロガー名（TensorBoardの場合）",
    )
    save_dir: str | None = Field(
        default=None,
        description="保存ディレクトリ（TensorBoardの場合）",
    )

    @field_validator("experiment_name")
    @classmethod
    def validate_experiment_name(cls, v: str) -> str:
        """experiment_nameが空でないことを検証."""
        if not v or not v.strip():
            raise ValueError("experiment_name は空にできません")
        return v


class LossConfig(BaseModel):
    """損失関数設定のバリデーションスキーマ."""

    _SUPPORTED_LOSS_TYPES = ["cross_entropy", "focal"]

    type: Literal["cross_entropy", "focal"] = Field(
        default="cross_entropy",
        description="損失関数タイプ",
    )
    weight: list[float] | Literal["balanced"] | None = Field(
        default=None,
        description="クラス重み（CrossEntropyLoss用）: 'balanced' または重みのリスト",
    )
    alpha: list[float] | Literal["balanced"] | None = Field(
        default=None,
        description="クラス重み（FocalLoss用）: 'balanced' または重みのリスト",
    )
    gamma: float = Field(
        default=2.0,
        ge=0.0,
        description="フォーカシングパラメータ（FocalLoss用）",
    )
    label_smoothing: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="ラベルスムージング係数",
    )

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """typeが有効な損失関数かを検証."""
        if v not in cls._SUPPORTED_LOSS_TYPES:
            raise ValueError(
                f"不明な損失関数タイプ: {v}. サポート: {cls._SUPPORTED_LOSS_TYPES}"
            )
        return v

    @model_validator(mode="after")
    def validate_loss_params(self) -> "LossConfig":
        """損失関数タイプに応じたパラメータを検証."""
        if self.type == "cross_entropy" and self.alpha is not None:
            raise ValueError(
                "alpha は FocalLoss 用のパラメータです。weight を使用してください。"
            )
        if self.type == "focal" and self.weight is not None:
            raise ValueError(
                "weight は CrossEntropyLoss 用のパラメータです。"
                "alpha を使用してください。"
            )
        return self


class TransformOpConfig(BaseModel):
    """単一トランスフォーム設定のバリデーションスキーマ."""

    type: str = Field(
        ...,
        description="トランスフォームのクラス名",
    )

    model_config = {"extra": "allow"}

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """typeが有効なトランスフォームかを検証."""
        if not v or not v.strip():
            raise ValueError("type は空にできません")

        if v == "ToTensorV2":
            return v

        if not hasattr(A, v):
            raise ValueError(
                f"不明なトランスフォームタイプ: {v}. "
                "albumentationsでサポートされているトランスフォームを指定してください"
            )
        return v


class TransformConfig(BaseModel):
    """トランスフォーム設定のバリデーションスキーマ."""

    ops: list[TransformOpConfig] = Field(
        default_factory=list,
        description="トランスフォーム操作のリスト",
    )

    @model_validator(mode="before")
    @classmethod
    def parse_ops(cls, data: Any) -> Any:
        """opsを TransformOpConfig のリストに変換."""
        if isinstance(data, dict) and "ops" in data:
            ops = data.get("ops", [])
            if isinstance(ops, list):
                data["ops"] = [
                    TransformOpConfig(**op) if isinstance(op, dict) else op
                    for op in ops
                ]
        return data


class ConfigValidationError(Exception):
    """設定バリデーションエラー."""

    def __init__(self, section: str, errors: list[Any]) -> None:
        """初期化.

        Args:
            section: エラーが発生した設定セクション名
            errors: バリデーションエラーのリスト
        """
        self.section = section
        self.errors = errors
        self.message = self._format_error_message()
        super().__init__(self.message)

    def _format_error_message(self) -> str:
        """エラーメッセージをフォーマット."""
        lines = [f"\n設定エラー [{self.section}]:"]
        lines.append("=" * 50)

        for error in self.errors:
            field = ".".join(str(loc) for loc in error.get("loc", []))
            msg = error.get("msg", "不明なエラー")
            input_val = error.get("input", "N/A")

            lines.append(f"  フィールド: {field}")
            lines.append(f"    入力値: {input_val}")
            lines.append(f"    エラー: {msg}")
            lines.append("")

        return "\n".join(lines)


def validate_checkpoint_config(config: dict) -> CheckpointConfig:
    """チェックポイント設定をバリデーション.

    Args:
        config: チェックポイント設定の辞書

    Returns:
        バリデーション済みのCheckpointConfig

    Raises:
        ConfigValidationError: バリデーションエラーが発生した場合
    """
    try:
        return CheckpointConfig(**config)
    except ValidationError as e:
        raise ConfigValidationError("checkpoint", e.errors()) from e
    except Exception as e:
        raise ConfigValidationError(
            "checkpoint",
            [{"loc": [], "msg": str(e), "input": config}],
        ) from e


def validate_progress_bar_config(config: dict) -> ProgressBarConfig:
    """プログレスバー設定をバリデーション.

    Args:
        config: プログレスバー設定の辞書

    Returns:
        バリデーション済みのProgressBarConfig

    Raises:
        ConfigValidationError: バリデーションエラーが発生した場合
    """
    try:
        return ProgressBarConfig(**config)
    except ValidationError as e:
        raise ConfigValidationError("progress_bar", e.errors()) from e
    except Exception as e:
        raise ConfigValidationError(
            "progress_bar",
            [{"loc": [], "msg": str(e), "input": config}],
        ) from e


def validate_trainer_config(config: dict) -> TrainerConfig:
    """トレーナー設定をバリデーション.

    Args:
        config: トレーナー設定の辞書

    Returns:
        バリデーション済みのTrainerConfig

    Raises:
        ConfigValidationError: バリデーションエラーが発生した場合
    """
    try:
        return TrainerConfig(**config)
    except ValidationError as e:
        raise ConfigValidationError("trainer", e.errors()) from e
    except Exception as e:
        raise ConfigValidationError(
            "trainer",
            [{"loc": [], "msg": str(e), "input": config}],
        ) from e


def validate_checkpoint_from_config(config: dict) -> CheckpointConfig:
    """全体設定辞書からチェックポイント設定をバリデーション.

    Args:
        config: 全体の設定辞書

    Returns:
        バリデーション済みのCheckpointConfig

    Raises:
        ConfigValidationError: バリデーションエラーが発生した場合
    """
    return validate_checkpoint_config(config["checkpoint"])


def validate_progress_bar_from_config(config: dict) -> ProgressBarConfig:
    """全体設定辞書からプログレスバー設定をバリデーション.

    Args:
        config: 全体の設定辞書

    Returns:
        バリデーション済みのProgressBarConfig

    Raises:
        ConfigValidationError: バリデーションエラーが発生した場合
    """
    return validate_progress_bar_config(config["progress_bar"])


def validate_trainer_from_config(config: dict) -> TrainerConfig:
    """全体設定辞書からトレーナー設定をバリデーション.

    Args:
        config: 全体の設定辞書

    Returns:
        バリデーション済みのTrainerConfig

    Raises:
        ConfigValidationError: バリデーションエラーが発生した場合
    """
    return validate_trainer_config(config["trainer"])


def validate_logger_config(config: dict) -> LoggerConfig:
    """ロガー設定をバリデーション.

    Args:
        config: ロガー設定の辞書

    Returns:
        バリデーション済みのLoggerConfig

    Raises:
        ConfigValidationError: バリデーションエラーが発生した場合
    """
    try:
        return LoggerConfig(**config)
    except ValidationError as e:
        raise ConfigValidationError("logger", e.errors()) from e
    except Exception as e:
        raise ConfigValidationError(
            "logger",
            [{"loc": [], "msg": str(e), "input": config}],
        ) from e


def validate_loss_config(config: dict) -> LossConfig:
    """損失関数設定をバリデーション.

    Args:
        config: 損失関数設定の辞書

    Returns:
        バリデーション済みのLossConfig

    Raises:
        ConfigValidationError: バリデーションエラーが発生した場合
    """
    try:
        return LossConfig(**config)
    except ValidationError as e:
        raise ConfigValidationError("loss", e.errors()) from e
    except Exception as e:
        raise ConfigValidationError(
            "loss",
            [{"loc": [], "msg": str(e), "input": config}],
        ) from e


def validate_loss_from_config(config: dict) -> LossConfig:
    """全体設定辞書から損失関数設定をバリデーション.

    Args:
        config: 全体の設定辞書

    Returns:
        バリデーション済みのLossConfig

    Raises:
        ConfigValidationError: バリデーションエラーが発生した場合
    """
    return validate_loss_config(config["loss"])


def validate_transform_config(config: dict) -> TransformConfig:
    """トランスフォーム設定をバリデーション.

    Args:
        config: トランスフォーム設定の辞書

    Returns:
        バリデーション済みのTransformConfig

    Raises:
        ConfigValidationError: バリデーションエラーが発生した場合
    """
    try:
        return TransformConfig(**config)
    except ValidationError as e:
        raise ConfigValidationError("augmentation", e.errors()) from e
    except Exception as e:
        raise ConfigValidationError(
            "augmentation",
            [{"loc": [], "msg": str(e), "input": config}],
        ) from e
