from typing import Any

from pytorch_lightning.loggers import Logger, MLFlowLogger, TensorBoardLogger

from config import (
    ConfigValidationError,
    LoggerConfig,
    validate_logger_config,
)


def build_logger(config: dict[str, Any], output_dir: str = "./outputs") -> Logger:
    """YAML設定からロガーを構築する.

    Args:
        config: ロガー設定辞書
            例: {"type": "mlflow", "experiment_name": "classification"}
        output_dir: 出力ディレクトリ（TensorBoardの場合に使用）

    Returns:
        pytorch_lightning.loggers.Logger オブジェクト

    Raises:
        ConfigValidationError: 設定のバリデーションに失敗した場合

    Example:
        >>> config = {"type": "mlflow", "experiment_name": "classification"}
        >>> logger = build_logger(config)
    """
    logger_cfg = validate_logger_config(config)

    if logger_cfg.type == "mlflow":
        return _build_mlflow_logger(logger_cfg)
    else:
        return _build_tensorboard_logger(logger_cfg, output_dir)


def _build_mlflow_logger(config: LoggerConfig) -> MLFlowLogger:
    """MLFlowLoggerを構築する.

    Args:
        config: バリデーション済みのLoggerConfig

    Returns:
        MLFlowLogger オブジェクト

    Raises:
        ConfigValidationError: ロガーの構築に失敗した場合
    """
    try:
        return MLFlowLogger(
            experiment_name=config.experiment_name,
            tracking_uri=config.tracking_uri,
            log_model=config.log_model,
        )
    except Exception as e:
        raise ConfigValidationError(
            section="logger",
            errors=[
                {
                    "loc": ["mlflow"],
                    "msg": str(e),
                    "input": {
                        "experiment_name": config.experiment_name,
                        "tracking_uri": config.tracking_uri,
                        "log_model": config.log_model,
                    },
                }
            ],
        ) from e


def _build_tensorboard_logger(
    config: LoggerConfig, output_dir: str
) -> TensorBoardLogger:
    """TensorBoardLoggerを構築する.

    Args:
        config: バリデーション済みのLoggerConfig
        output_dir: 出力ディレクトリ

    Returns:
        TensorBoardLogger オブジェクト

    Raises:
        ConfigValidationError: ロガーの構築に失敗した場合
    """
    save_dir = config.save_dir if config.save_dir else output_dir
    try:
        return TensorBoardLogger(
            save_dir=save_dir,
            name=config.name,
        )
    except Exception as e:
        raise ConfigValidationError(
            section="logger",
            errors=[
                {
                    "loc": ["tensorboard"],
                    "msg": str(e),
                    "input": {"save_dir": save_dir, "name": config.name},
                }
            ],
        ) from e
