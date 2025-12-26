"""画像分類学習のエントリーポイント.

YAML設定ファイルを使用して学習を実行する。
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import pytorch_lightning as L
from loguru import logger
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import Logger as PLLogger

from builders import build_logger
from config import load_config, override_config, save_config
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
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only run test",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )

    # 残りの引数はオーバーライドとして扱う
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
    """
    trainer_config = config.get("trainer", {})
    checkpoint_config = config.get("checkpoint", {})

    # チェックポイントコールバック
    checkpoint_callback = ModelCheckpoint(
        dirpath=exp_dir / "checkpoints",
        filename="{epoch:02d}-{val_loss:.4f}",
        monitor=checkpoint_config.get("monitor", "val_loss"),
        mode=checkpoint_config.get("mode", "min"),
        save_top_k=checkpoint_config.get("save_top_k", 3),
        save_last=checkpoint_config.get("save_last", True),
    )

    # コールバック
    callbacks = [
        RichProgressBar(),
        checkpoint_callback,
    ]

    # Trainer
    trainer = L.Trainer(
        logger=pl_logger,
        callbacks=callbacks,
        max_epochs=trainer_config.get("max_epochs", 100),
        accelerator=trainer_config.get("accelerator", "gpu"),
        devices=trainer_config.get("devices", 1),
        precision=trainer_config.get("precision", "16-mixed"),
        strategy=trainer_config.get("strategy", "auto"),
        deterministic=trainer_config.get("deterministic", True),
        log_every_n_steps=trainer_config.get("log_every_n_steps", 10),
        val_check_interval=trainer_config.get("val_check_interval", 1.0),
        num_sanity_val_steps=0,
    )

    return trainer


def main() -> None:
    """メイン関数."""
    args = parse_args()

    # 設定を読み込み
    logger.info(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # コマンドラインオーバーライドを適用
    if args.overrides:
        config = override_config(config, args.overrides)

    # シードのオーバーライド
    if args.seed is not None:
        config["seed"] = args.seed

    # シードを設定
    seed = config.get("seed", 42)
    L.seed_everything(seed)
    logger.info(f"Random seed: {seed}")

    # 出力ディレクトリをセットアップ
    exp_dir = setup_output_dir(config)
    logger.info(f"Experiment directory: {exp_dir}")

    # 使用した設定を保存
    save_config(config, exp_dir / "config.yaml")

    # ロガーを構築
    output_dir = config.get("output_dir", "./outputs")
    pl_logger = build_logger(config.get("logger", {}), output_dir)

    # ハイパーパラメータをログ
    if hasattr(pl_logger, "log_hyperparams"):
        pl_logger.log_hyperparams(config)

    # DataModuleを構築
    logger.info("Building DataModule...")
    datamodule = ClassificationDataModule(config)

    # モデルを構築
    logger.info("Building model...")
    model = ImageClassifier(config)

    # Trainerを構築
    logger.info("Building Trainer...")
    trainer = build_trainer(config, exp_dir, pl_logger)

    if args.test_only:
        # テストのみ実行
        if args.resume is None:
            logger.error("--resume is required for --test-only")
            sys.exit(1)

        logger.info(f"Running test with checkpoint: {args.resume}")
        trainer.test(model, datamodule=datamodule, ckpt_path=args.resume)
    else:
        # 学習を実行
        logger.info("Starting training...")
        trainer.fit(
            model,
            datamodule=datamodule,
            ckpt_path=args.resume,
        )

        # テストを実行
        logger.info("Running test...")
        trainer.test(model, datamodule=datamodule, ckpt_path="best")

    logger.info("Done!")


if __name__ == "__main__":
    main()
