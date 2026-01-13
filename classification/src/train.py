import argparse
from pathlib import Path
from typing import Any

import pytorch_lightning as L
import torch
from loguru import logger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import Logger as PLLogger
from pytorch_lightning.loggers import MLFlowLogger

from builders import build_logger
from config import (
    ConfigValidationError,
    load_config,
    save_config,
    validate_checkpoint_from_config,
    validate_progress_bar_from_config,
    validate_trainer_from_config,
)
from data import ClassificationDataModule
from models import ImageClassifier
from validate import run_validation


def setup_output_dir(config: dict[str, Any], run_id: str | None = None) -> Path:
    """出力ディレクトリをセットアップする.

    Args:
        config: 設定辞書
        run_id: MLflowのrun_id（指定された場合はディレクトリ名に含める）

    Returns:
        出力ディレクトリのパス
    """
    output_dir = Path(config.get("output_dir", "./outputs"))
    exp_name = config.get("exp_name", "default")

    if run_id:
        exp_dir = output_dir / exp_name / run_id
    else:
        exp_dir = output_dir / exp_name

    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "checkpoints").mkdir(exist_ok=True)

    return exp_dir


def build_trainer(
    config: dict[str, Any],
    exp_dir: Path,
    pl_logger: PLLogger,
) -> L.Trainer:
    """PyTorch Lightning Trainerを構築する.

    Args:
        config: 設定辞書
        exp_dir: 実験ディレクトリ
        pl_logger: PyTorch Lightningロガー

    Returns:
        Trainerオブジェクト

    Raises:
        ConfigValidationError: 設定のバリデーションに失敗した場合
    """
    checkpoint_cfg = validate_checkpoint_from_config(config)
    try:
        checkpoint_callback = ModelCheckpoint(
            dirpath=exp_dir / "checkpoints",
            filename="best",
            monitor=checkpoint_cfg.monitor,
            mode=checkpoint_cfg.mode,
            save_top_k=checkpoint_cfg.save_top_k,
            save_last=checkpoint_cfg.save_last,
        )
    except Exception as e:
        raise ConfigValidationError(
            section="checkpoint",
            errors=[{"loc": [], "msg": str(e), "input": "N/A"}],
        ) from e

    progress_bar_cfg = validate_progress_bar_from_config(config)
    try:
        tqdm_callback = TQDMProgressBar(refresh_rate=progress_bar_cfg.refresh_rate)
    except Exception as e:
        raise ConfigValidationError(
            section="progress_bar",
            errors=[
                {
                    "loc": ["refresh_rate"],
                    "msg": str(e),
                    "input": progress_bar_cfg.refresh_rate,
                }
            ],
        ) from e

    trainer_cfg = validate_trainer_from_config(config)
    try:
        trainer = L.Trainer(
            logger=pl_logger,
            callbacks=[
                tqdm_callback,
                checkpoint_callback,
            ],
            max_epochs=trainer_cfg.max_epochs,
            accelerator=trainer_cfg.accelerator,
            devices=trainer_cfg.devices,
            precision=trainer_cfg.precision,
            strategy=trainer_cfg.strategy,
            deterministic=trainer_cfg.deterministic,
            log_every_n_steps=trainer_cfg.log_every_n_steps,
            val_check_interval=trainer_cfg.val_check_interval,
            num_sanity_val_steps=0,
        )
    except Exception as e:
        raise ConfigValidationError(
            section="trainer",
            errors=[{"loc": [], "msg": str(e), "input": "N/A"}],
        ) from e

    return trainer


def run_training(
    config_path: str,
    resume: str | None = None,
    validate: bool = False,
) -> None:
    """トレーニングを実行する.

    Args:
        config_path: 設定ファイルのパス
        resume: 再開するチェックポイントのパス
        validate: トレーニング後に検証を実行するかどうか
    """
    torch.set_float32_matmul_precision("high")

    config = load_config(config_path)

    seed = config["seed"]
    L.seed_everything(seed, workers=True)

    output_dir = config.get("output_dir", "./outputs")
    pl_logger = build_logger(config.get("logger", {}), output_dir)

    run_id = None
    if isinstance(pl_logger, MLFlowLogger):
        run_id = pl_logger.run_id
        logger.info(f"MLflow run_id: {run_id}")

    exp_dir = setup_output_dir(config, run_id)
    logger.info(f"Experiment directory: {exp_dir}")

    config_path_saved = exp_dir / "config.yaml"
    save_config(config, config_path_saved)

    if isinstance(pl_logger, MLFlowLogger):
        pl_logger.experiment.log_artifact(pl_logger.run_id, str(config_path_saved))
        logger.info("Saved merged config as MLflow artifact")

    if hasattr(pl_logger, "log_hyperparams"):
        pl_logger.log_hyperparams(config)

    logger.info("Building DataModule...")
    datamodule = ClassificationDataModule(config)

    logger.info("Building model...")
    model = ImageClassifier(config)

    logger.info("Building Trainer...")
    trainer = build_trainer(config, exp_dir, pl_logger)

    logger.info("Starting training...")
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=resume,
    )

    logger.info("Training done!")

    if validate and run_id:
        logger.info("=" * 50)
        logger.info("Starting validation...")
        logger.info("=" * 50)

        run_validation(config_path=config_path, run_id=run_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Classification Training")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation after training",
    )
    args = parser.parse_args()

    run_training(
        config_path=args.config,
        resume=args.resume,
        validate=args.validate,
    )
