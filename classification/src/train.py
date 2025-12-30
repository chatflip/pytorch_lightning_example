import argparse
from pathlib import Path
from typing import Any

import pytorch_lightning as L
from loguru import logger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import Logger as PLLogger

from builders import build_logger
from config import (
    ConfigValidationError,
    load_config,
    override_config,
    save_config,
    validate_training_configs,
)
from data import ClassificationDataModule
from models import ImageClassifier


def parse_args() -> argparse.Namespace:
    """コマンドライン引数をパースする."""
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
    args, unknown = parser.parse_known_args()
    args.overrides = unknown
    return args


def setup_output_dir(config: dict[str, Any]) -> Path:
    """出力ディレクトリをセットアップする.

    Args:
        config: 設定辞書

    Returns:
        出力ディレクトリのパス
    """
    output_dir = Path(config.get("output_dir", "./outputs"))
    exp_name = config.get("exp_name", "default")
    exp_dir = output_dir / exp_name

    # ディレクトリを作成
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
    checkpoint_cfg, progress_bar_cfg, trainer_cfg = validate_training_configs(config)

    checkpoint_callback = ModelCheckpoint(
        dirpath=exp_dir / "checkpoints",
        filename="best",
        monitor=checkpoint_cfg.monitor,
        mode=checkpoint_cfg.mode,
        save_top_k=checkpoint_cfg.save_top_k,
        save_last=checkpoint_cfg.save_last,
    )

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

    callbacks = [
        tqdm_callback,
        checkpoint_callback,
    ]

    try:
        trainer = L.Trainer(
            logger=pl_logger,
            callbacks=callbacks,
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


def main() -> None:
    """メイン関数."""
    args = parse_args()

    logger.info(f"Loading config from: {args.config}")
    config = load_config(args.config)

    if args.overrides:
        config = override_config(config, args.overrides)

    seed = config.get("seed", 42)
    L.seed_everything(seed)
    logger.info(f"Random seed: {seed}")

    exp_dir = setup_output_dir(config)
    logger.info(f"Experiment directory: {exp_dir}")

    save_config(config, exp_dir / "config.yaml")

    output_dir = config.get("output_dir", "./outputs")
    pl_logger = build_logger(config.get("logger", {}), output_dir)

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
        ckpt_path=args.resume,
    )

    logger.info("Running test...")
    trainer.test(model, datamodule=datamodule, ckpt_path="best")

    logger.info("Done!")


if __name__ == "__main__":
    main()
