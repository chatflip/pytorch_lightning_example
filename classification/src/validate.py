import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as L
import seaborn as sns
import torch
import torch.nn.functional as F
from loguru import logger
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix
from tqdm import tqdm

from config import load_config
from data import ClassificationDataModule
from models import ImageClassifier


def parse_args() -> argparse.Namespace:
    """コマンドライン引数をパースする."""
    parser = argparse.ArgumentParser(description="Validation Script")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (auto-detected if not specified)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: {exp_dir}/validation)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference",
    )

    return parser.parse_args()


def find_best_checkpoint(checkpoint_dir: Path) -> Path:
    """チェックポイントディレクトリから最良のチェックポイントを見つける."""
    # best.ckptがあればそれを使用
    best_ckpt = checkpoint_dir / "best.ckpt"
    if best_ckpt.exists():
        return best_ckpt

    # なければlatest.ckptを使用
    latest_ckpt = checkpoint_dir / "latest.ckpt"
    if latest_ckpt.exists():
        return latest_ckpt

    raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")


def validate(
    model: ImageClassifier,
    datamodule: ClassificationDataModule,
    device: torch.device,
    num_classes: int,
) -> tuple[list[dict[str, Any]], dict[str, float], np.ndarray]:
    """Validationを実行する.

    Args:
        model: 学習済みモデル
        datamodule: データモジュール
        device: 推論に使用するデバイス
        num_classes: クラス数

    Returns:
        predictions: 各画像の予測結果のリスト
        metrics: メトリクスの辞書
        confusion_matrix: 混同行列
    """
    model.eval()
    model.to(device)

    # メトリクスを初期化
    acc1 = MulticlassAccuracy(num_classes=num_classes, top_k=1).to(device)
    acc5 = MulticlassAccuracy(num_classes=num_classes, top_k=5).to(device)
    confusion = MulticlassConfusionMatrix(num_classes=num_classes).to(device)

    # 結果を格納するリスト
    predictions: list[dict[str, Any]] = []
    total_loss = 0.0
    total_samples = 0

    # データセットから画像パスを取得するための準備
    val_dataset = datamodule.val_dataset
    if val_dataset is None:
        raise RuntimeError("val_dataset is not initialized")

    samples = val_dataset.samples  # [(path, label), ...]
    classes = val_dataset.classes

    # バッチ処理
    val_loader = datamodule.val_dataloader()
    sample_idx = 0

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            targets = targets.to(device)
            batch_size = images.size(0)

            # 推論
            outputs = model(images)
            loss = F.cross_entropy(outputs, targets, reduction="sum")

            # メトリクスを更新
            acc1.update(outputs, targets)
            acc5.update(outputs, targets)
            confusion.update(outputs, targets)

            total_loss += loss.item()
            total_samples += batch_size

            # 各画像の予測結果を収集
            probs = F.softmax(outputs, dim=1)
            pred_labels = outputs.argmax(dim=1)
            pred_probs, _ = probs.max(dim=1)

            for i in range(batch_size):
                img_path, gt_label = samples[sample_idx]
                pred_label = pred_labels[i].item()
                pred_prob = pred_probs[i].item()

                predictions.append(
                    {
                        "image_path": str(img_path),
                        "ground_truth": classes[gt_label],
                        "prediction": classes[pred_label],
                        "gt_label_idx": gt_label,
                        "pred_label_idx": pred_label,
                        "confidence": pred_prob,
                        "correct": gt_label == pred_label,
                    }
                )
                sample_idx += 1

    # メトリクスを計算
    metrics = {
        "loss": total_loss / total_samples,
        "top1_accuracy": acc1.compute().item(),
        "top5_accuracy": acc5.compute().item(),
        "total_samples": total_samples,
    }

    # 混同行列を取得
    confusion_matrix = confusion.compute().cpu().numpy()

    return predictions, metrics, confusion_matrix


def save_predictions(
    predictions: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """予測結果をファイルに保存する.

    Args:
        predictions: 予測結果のリスト
        output_path: 出力ファイルパス
    """
    with open(output_path, "w", encoding="utf-8") as f:
        # ヘッダー
        f.write("image_path\tground_truth\tprediction\tconfidence\tcorrect\n")
        # 各画像の結果
        for pred in predictions:
            line = (
                f"{pred['image_path']}\t"
                f"{pred['ground_truth']}\t"
                f"{pred['prediction']}\t"
                f"{pred['confidence']:.4f}\t"
                f"{pred['correct']}\n"
            )
            f.write(line)

    logger.info(f"Predictions saved to: {output_path}")


def save_metrics(metrics: dict[str, float], output_path: Path) -> None:
    """メトリクスをファイルに保存する.

    Args:
        metrics: メトリクスの辞書
        output_path: 出力ファイルパス
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=== Validation Metrics ===\n\n")
        f.write(f"Total Samples: {metrics['total_samples']}\n")
        f.write(f"Loss: {metrics['loss']:.4f}\n")
        top1 = metrics["top1_accuracy"]
        top5 = metrics["top5_accuracy"]
        f.write(f"Top-1 Accuracy: {top1:.4f} ({top1 * 100:.2f}%)\n")
        f.write(f"Top-5 Accuracy: {top5:.4f} ({top5 * 100:.2f}%)\n")

    logger.info(f"Metrics saved to: {output_path}")


def save_confusion_matrix(
    confusion_matrix: np.ndarray,
    classes: list[str],
    output_path: Path,
    max_classes_to_show: int = 30,
) -> None:
    """混同行列を画像として保存する.

    Args:
        confusion_matrix: 混同行列
        classes: クラス名のリスト
        output_path: 出力ファイルパス
        max_classes_to_show: 表示する最大クラス数
    """
    num_classes = len(classes)

    # クラス数が多い場合は上位クラスのみ表示
    if num_classes > max_classes_to_show:
        logger.warning(
            f"Too many classes ({num_classes}). "
            f"Showing confusion matrix for top {max_classes_to_show} classes."
        )
        # 各クラスのサンプル数を計算
        class_counts = confusion_matrix.sum(axis=1)
        top_indices = np.argsort(class_counts)[-max_classes_to_show:]
        confusion_matrix = confusion_matrix[np.ix_(top_indices, top_indices)]
        classes = [classes[i] for i in top_indices]

    # 図のサイズを計算
    fig_size = max(10, len(classes) * 0.3)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # ヒートマップを描画
    sns.heatmap(
        confusion_matrix,
        annot=len(classes) <= 20,  # 20クラス以下なら数値を表示
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Confusion Matrix")

    # ラベルを回転
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"Confusion matrix saved to: {output_path}")


def main() -> None:
    """メイン関数."""
    args = parse_args()

    # 設定を読み込み
    logger.info(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # 実験ディレクトリを取得
    base_output_dir = Path(config.get("output_dir", "./outputs"))
    exp_name = config.get("exp_name", "default")
    exp_dir = base_output_dir / exp_name

    # チェックポイントを解決
    if args.checkpoint is None:
        checkpoint_dir = exp_dir / "checkpoints"
        checkpoint_path = find_best_checkpoint(checkpoint_dir)
        logger.info(f"Auto-detected checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = Path(args.checkpoint)

    # 出力ディレクトリを解決
    if args.output_dir is None:
        output_dir = exp_dir / "validation"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # シードを設定
    seed = config.get("seed", 42)
    L.seed_everything(seed)

    # デバイスを設定
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # DataModuleを構築
    logger.info("Building DataModule...")
    datamodule = ClassificationDataModule(config)
    datamodule.setup(stage="validate")

    # モデルを読み込み
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    model = ImageClassifier.load_from_checkpoint(checkpoint_path, config=config)

    # クラス情報を取得
    num_classes = config.get("data", {}).get("num_classes", datamodule.num_classes)
    classes = datamodule.classes
    logger.info(f"Number of classes: {num_classes}")

    # Validationを実行
    logger.info("Running validation...")
    predictions, metrics, confusion_matrix = validate(
        model=model,
        datamodule=datamodule,
        device=device,
        num_classes=num_classes,
    )

    # 結果を表示
    logger.info("=" * 50)
    logger.info("Validation Results:")
    logger.info(f"  Total Samples: {metrics['total_samples']}")
    logger.info(f"  Loss: {metrics['loss']:.4f}")
    logger.info(f"  Top-1 Accuracy: {metrics['top1_accuracy'] * 100:.2f}%")
    logger.info(f"  Top-5 Accuracy: {metrics['top5_accuracy'] * 100:.2f}%")
    logger.info("=" * 50)

    # 正解・不正解の統計
    correct_count = sum(1 for p in predictions if p["correct"])
    incorrect_count = len(predictions) - correct_count
    logger.info(f"  Correct: {correct_count}")
    logger.info(f"  Incorrect: {incorrect_count}")

    # ファイルに保存
    save_predictions(predictions, output_dir / "predictions.txt")
    save_metrics(metrics, output_dir / "metrics.txt")
    save_confusion_matrix(
        confusion_matrix, classes, output_dir / "confusion_matrix.png"
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()
