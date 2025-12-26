"""ロガービルダー.

YAML設定からPyTorch Lightningロガーを構築する。
"""

from typing import Any

from pytorch_lightning.loggers import Logger, MLFlowLogger, TensorBoardLogger


def build_logger(config: dict[str, Any], output_dir: str = "./outputs") -> Logger:
    """YAML設定からロガーを構築する.

    Args:
        config: ロガー設定辞書
            例: {"type": "mlflow", "experiment_name": "classification"}
        output_dir: 出力ディレクトリ（TensorBoardの場合に使用）

    Returns:
        pytorch_lightning.loggers.Logger オブジェクト

    Raises:
        ValueError: 未知のロガータイプの場合

    Example:
        >>> config = {"type": "mlflow", "experiment_name": "classification"}
        >>> logger = build_logger(config)
    """
    config = config.copy()
    logger_type = config.pop("type", "mlflow").lower()

    if logger_type == "mlflow":
        return _build_mlflow_logger(config)
    elif logger_type == "tensorboard":
        return _build_tensorboard_logger(config, output_dir)
    else:
        raise ValueError(
            f"Unknown logger type: {logger_type}. Supported: ['mlflow', 'tensorboard']"
        )


def _build_mlflow_logger(config: dict[str, Any]) -> MLFlowLogger:
    """MLFlowLoggerを構築する.

    Args:
        config: MLFlow設定辞書

    Returns:
        MLFlowLogger オブジェクト
    """
    experiment_name = config.get("experiment_name", "classification")
    tracking_uri = config.get("tracking_uri", None)
    log_model = config.get("log_model", True)

    return MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        log_model=log_model,
    )


def _build_tensorboard_logger(
    config: dict[str, Any], output_dir: str
) -> TensorBoardLogger:
    """TensorBoardLoggerを構築する.

    Args:
        config: TensorBoard設定辞書
        output_dir: 出力ディレクトリ

    Returns:
        TensorBoardLogger オブジェクト
    """
    name = config.get("name", "tensorboard")
    save_dir = config.get("save_dir", output_dir)

    return TensorBoardLogger(
        save_dir=save_dir,
        name=name,
    )
