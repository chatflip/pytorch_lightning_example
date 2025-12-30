import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as L
import seaborn as sns
import torch
import torch.nn.functional as F
from loguru import logger
from mlflow import log_artifact, log_metrics, set_tracking_uri, start_run
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from tqdm import tqdm

from config import load_config
from data import ClassificationDataModule
from models import ImageClassifier


def find_best_checkpoint(checkpoint_dir: Path) -> Path:
    """チェックポイントディレクトリからbest.ckptを見つける."""
    best_ckpt = checkpoint_dir / "best.ckpt"
    if best_ckpt.exists():
        return best_ckpt

    raise FileNotFoundError(f"best.ckpt not found in {checkpoint_dir}")


def validate(
    model: ImageClassifier,
    datamodule: ClassificationDataModule,
    device: torch.device,
    num_classes: int,
) -> tuple[list[dict[str, Any]], dict[str, float], np.ndarray, dict[str, np.ndarray]]:
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
        class_metrics: クラスごとのメトリクス（accuracy, precision, recall, f1）
    """
    model.eval()
    model.to(device)

    acc1 = MulticlassAccuracy(num_classes=num_classes, top_k=1).to(device)
    acc5 = MulticlassAccuracy(num_classes=num_classes, top_k=5).to(device)
    # クラスごとのメトリクス (average=None)
    acc_per_class = MulticlassAccuracy(
        num_classes=num_classes, top_k=1, average=None
    ).to(device)
    prec_per_class = MulticlassPrecision(num_classes=num_classes, average=None).to(
        device
    )
    recall_per_class = MulticlassRecall(num_classes=num_classes, average=None).to(
        device
    )
    f1_per_class = MulticlassF1Score(num_classes=num_classes, average=None).to(device)
    # Macro平均
    prec_macro = MulticlassPrecision(num_classes=num_classes, average="macro").to(
        device
    )
    recall_macro = MulticlassRecall(num_classes=num_classes, average="macro").to(device)
    f1_macro = MulticlassF1Score(num_classes=num_classes, average="macro").to(device)
    # Weighted平均
    prec_weighted = MulticlassPrecision(num_classes=num_classes, average="weighted").to(
        device
    )
    recall_weighted = MulticlassRecall(num_classes=num_classes, average="weighted").to(
        device
    )
    f1_weighted = MulticlassF1Score(num_classes=num_classes, average="weighted").to(
        device
    )
    confusion = MulticlassConfusionMatrix(num_classes=num_classes).to(device)

    predictions: list[dict[str, Any]] = []
    total_loss = 0.0
    total_samples = 0

    val_dataset = datamodule.val_dataset
    if val_dataset is None:
        raise RuntimeError("val_dataset is not initialized")

    samples = val_dataset.samples  # [(path, label), ...]
    classes = val_dataset.classes

    val_loader = datamodule.val_dataloader()
    sample_idx = 0

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            targets = targets.to(device)
            batch_size = images.size(0)

            outputs = model(images)
            loss = F.cross_entropy(outputs, targets, reduction="sum")

            acc1.update(outputs, targets)
            acc5.update(outputs, targets)
            acc_per_class.update(outputs, targets)
            prec_per_class.update(outputs, targets)
            recall_per_class.update(outputs, targets)
            f1_per_class.update(outputs, targets)
            prec_macro.update(outputs, targets)
            recall_macro.update(outputs, targets)
            f1_macro.update(outputs, targets)
            prec_weighted.update(outputs, targets)
            recall_weighted.update(outputs, targets)
            f1_weighted.update(outputs, targets)
            confusion.update(outputs, targets)

            total_loss += loss.item()
            total_samples += batch_size

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

    metrics = {
        "loss": total_loss / total_samples,
        "top1_accuracy": acc1.compute().item(),  # type: ignore[call-arg]
        "top5_accuracy": acc5.compute().item(),  # type: ignore[call-arg]
        "total_samples": total_samples,
        "precision_macro": prec_macro.compute().item(),  # type: ignore[call-arg]
        "recall_macro": recall_macro.compute().item(),  # type: ignore[call-arg]
        "f1_macro": f1_macro.compute().item(),  # type: ignore[call-arg]
        "precision_weighted": prec_weighted.compute().item(),  # type: ignore[call-arg]
        "recall_weighted": recall_weighted.compute().item(),  # type: ignore[call-arg]
        "f1_weighted": f1_weighted.compute().item(),  # type: ignore[call-arg]
    }

    confusion_matrix = confusion.compute().cpu().numpy()  # type: ignore[call-arg]

    class_metrics = {
        "accuracy": acc_per_class.compute().cpu().numpy(),  # type: ignore[call-arg]
        "precision": prec_per_class.compute().cpu().numpy(),  # type: ignore[call-arg]
        "recall": recall_per_class.compute().cpu().numpy(),  # type: ignore[call-arg]
        "f1": f1_per_class.compute().cpu().numpy(),  # type: ignore[call-arg]
    }

    return predictions, metrics, confusion_matrix, class_metrics


def save_predictions(
    predictions: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """予測結果をCSVファイルに保存する.

    Args:
        predictions: 予測結果のリスト
        output_path: 出力ファイルパス
    """
    fieldnames = ["image_path", "ground_truth", "prediction", "confidence", "correct"]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for pred in predictions:
            writer.writerow(
                {
                    "image_path": pred["image_path"],
                    "ground_truth": pred["ground_truth"],
                    "prediction": pred["prediction"],
                    "confidence": f"{pred['confidence']:.4f}",
                    "correct": pred["correct"],
                }
            )

    logger.info(f"Predictions saved to: {output_path}")


def save_metrics(metrics: dict[str, float], output_path: Path) -> None:
    """メトリクスをCSVファイルに保存する.

    Args:
        metrics: メトリクスの辞書
        output_path: 出力ファイルパス
    """
    fieldnames = ["metric", "value"]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"metric": "total_samples", "value": metrics["total_samples"]})
        writer.writerow({"metric": "loss", "value": f"{metrics['loss']:.4f}"})
        writer.writerow(
            {"metric": "top1_accuracy", "value": f"{metrics['top1_accuracy']:.4f}"}
        )
        writer.writerow(
            {"metric": "top5_accuracy", "value": f"{metrics['top5_accuracy']:.4f}"}
        )
        # Macro平均
        writer.writerow(
            {"metric": "precision_macro", "value": f"{metrics['precision_macro']:.4f}"}
        )
        writer.writerow(
            {"metric": "recall_macro", "value": f"{metrics['recall_macro']:.4f}"}
        )
        writer.writerow({"metric": "f1_macro", "value": f"{metrics['f1_macro']:.4f}"})
        # Weighted平均
        writer.writerow(
            {
                "metric": "precision_weighted",
                "value": f"{metrics['precision_weighted']:.4f}",
            }
        )
        writer.writerow(
            {
                "metric": "recall_weighted",
                "value": f"{metrics['recall_weighted']:.4f}",
            }
        )
        writer.writerow(
            {"metric": "f1_weighted", "value": f"{metrics['f1_weighted']:.4f}"}
        )

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

    if num_classes > max_classes_to_show:
        logger.warning(
            f"Too many classes ({num_classes}). "
            f"Showing confusion matrix for top {max_classes_to_show} classes."
        )
        class_counts = confusion_matrix.sum(axis=1)
        top_indices = np.argsort(class_counts)[-max_classes_to_show:]
        confusion_matrix = confusion_matrix[np.ix_(top_indices, top_indices)]
        classes = [classes[i] for i in top_indices]

    fig_size = max(10, len(classes) * 0.3)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

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

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"Confusion matrix saved to: {output_path}")


def save_class_metrics(
    class_metrics: dict[str, np.ndarray],
    classes: list[str],
    class_samples: np.ndarray,
    class_correct: np.ndarray,
    output_path: Path,
) -> None:
    """クラスごとのメトリクスをCSVファイルに保存する.

    Args:
        class_metrics: クラスごとのメトリクス（accuracy, precision, recall, f1）
        classes: クラス名のリスト
        class_samples: クラスごとのサンプル数
        class_correct: クラスごとの正解数
        output_path: 出力ファイルパス
    """
    fieldnames = [
        "class_name",
        "samples",
        "correct",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, class_name in enumerate(classes):
            writer.writerow(
                {
                    "class_name": class_name,
                    "samples": int(class_samples[i]),
                    "correct": int(class_correct[i]),
                    "accuracy": f"{class_metrics['accuracy'][i]:.4f}",
                    "precision": f"{class_metrics['precision'][i]:.4f}",
                    "recall": f"{class_metrics['recall'][i]:.4f}",
                    "f1": f"{class_metrics['f1'][i]:.4f}",
                }
            )

    logger.info(f"Class metrics saved to: {output_path}")


def main(args: argparse.Namespace) -> None:
    """メイン関数.

    Args:
        args: コマンドライン引数
    """
    logger.info(f"Loading config from: {args.config}")
    config = load_config(args.config)

    base_output_dir = Path(config.get("output_dir", "./outputs"))
    exp_name = config.get("exp_name", "default")

    if args.checkpoint is not None:
        checkpoint_path = Path(args.checkpoint)
        exp_dir = checkpoint_path.parent.parent
        logger.info(f"Using checkpoint: {checkpoint_path}")
    elif args.run_id is not None:
        exp_dir = base_output_dir / exp_name / args.run_id
        checkpoint_dir = exp_dir / "checkpoints"
        checkpoint_path = find_best_checkpoint(checkpoint_dir)
        logger.info(f"Auto-detected checkpoint: {checkpoint_path}")
    else:
        raise ValueError(
            "Either --checkpoint or --run-id must be specified. "
            "Use --run-id to specify the MLflow run_id for the experiment."
        )

    if args.output_dir is None:
        output_dir = exp_dir / "validation"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    seed = config.get("seed", 42)
    L.seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Building DataModule...")
    datamodule = ClassificationDataModule(config)
    datamodule.setup(stage="validate")

    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    model = ImageClassifier.load_from_checkpoint(checkpoint_path, config=config)

    num_classes = config.get("data", {}).get("num_classes", datamodule.num_classes)
    classes = datamodule.classes
    logger.info(f"Number of classes: {num_classes}")

    logger.info("Running validation...")
    predictions, metrics, confusion_matrix, class_metrics = validate(
        model=model,
        datamodule=datamodule,
        device=device,
        num_classes=num_classes,
    )

    logger.info("=" * 50)
    logger.info("Validation Results:")
    logger.info(f"  Total Samples: {metrics['total_samples']}")
    logger.info(f"  Loss: {metrics['loss']:.4f}")
    logger.info(f"  Top-1 Accuracy: {metrics['top1_accuracy'] * 100:.2f}%")
    logger.info(f"  Top-5 Accuracy: {metrics['top5_accuracy'] * 100:.2f}%")
    logger.info("=" * 50)

    correct_count = sum(1 for p in predictions if p["correct"])
    incorrect_count = len(predictions) - correct_count
    logger.info(f"  Correct: {correct_count}")
    logger.info(f"  Incorrect: {incorrect_count}")

    logger.info("=" * 50)
    logger.info("Macro Average:")
    logger.info(f"  Precision: {metrics['precision_macro'] * 100:.2f}%")
    logger.info(f"  Recall:    {metrics['recall_macro'] * 100:.2f}%")
    logger.info(f"  F1-score:  {metrics['f1_macro'] * 100:.2f}%")
    logger.info("Weighted Average:")
    logger.info(f"  Precision: {metrics['precision_weighted'] * 100:.2f}%")
    logger.info(f"  Recall:    {metrics['recall_weighted'] * 100:.2f}%")
    logger.info(f"  F1-score:  {metrics['f1_weighted'] * 100:.2f}%")
    logger.info("=" * 50)

    class_samples = confusion_matrix.sum(axis=1)
    class_correct = np.diag(confusion_matrix)

    logger.info("=" * 50)
    logger.info("Per-Class Metrics (sorted by F1-score):")
    sorted_indices = np.argsort(class_metrics["f1"])

    logger.info("  [Worst 10 classes]")
    for i, idx in enumerate(sorted_indices[:10]):
        logger.info(
            f"    {i + 1:2d}. {classes[idx]}: "
            f"F1={class_metrics['f1'][idx] * 100:.1f}% "
            f"P={class_metrics['precision'][idx] * 100:.1f}% "
            f"R={class_metrics['recall'][idx] * 100:.1f}% "
            f"({int(class_correct[idx])}/{int(class_samples[idx])})"
        )

    logger.info("  [Best 10 classes]")
    for i, idx in enumerate(reversed(sorted_indices[-10:])):
        logger.info(
            f"    {i + 1:2d}. {classes[idx]}: "
            f"F1={class_metrics['f1'][idx] * 100:.1f}% "
            f"P={class_metrics['precision'][idx] * 100:.1f}% "
            f"R={class_metrics['recall'][idx] * 100:.1f}% "
            f"({int(class_correct[idx])}/{int(class_samples[idx])})"
        )
    logger.info("=" * 50)

    predictions_path = output_dir / "predictions.csv"
    metrics_path = output_dir / "metrics.csv"
    confusion_matrix_path = output_dir / "confusion_matrix.png"
    class_metrics_path = output_dir / "class_metrics.csv"

    save_predictions(predictions, predictions_path)
    save_metrics(metrics, metrics_path)
    save_confusion_matrix(confusion_matrix, classes, confusion_matrix_path)
    save_class_metrics(
        class_metrics, classes, class_samples, class_correct, class_metrics_path
    )

    if args.run_id is not None:
        logger.info(f"Logging artifacts to MLflow run: {args.run_id}")
        logger_config = config.get("logger", {})
        tracking_uri = logger_config.get("tracking_uri", "mlruns")
        set_tracking_uri(tracking_uri)

        with start_run(run_id=args.run_id):
            log_artifact(str(predictions_path), artifact_path="validation")
            log_artifact(str(metrics_path), artifact_path="validation")
            log_artifact(str(confusion_matrix_path), artifact_path="validation")
            log_artifact(str(class_metrics_path), artifact_path="validation")
            log_metrics(
                {
                    "val_loss": metrics["loss"],
                    "val_top1_accuracy": metrics["top1_accuracy"],
                    "val_top5_accuracy": metrics["top5_accuracy"],
                    "val_precision_macro": metrics["precision_macro"],
                    "val_recall_macro": metrics["recall_macro"],
                    "val_f1_macro": metrics["f1_macro"],
                    "val_precision_weighted": metrics["precision_weighted"],
                    "val_recall_weighted": metrics["recall_weighted"],
                    "val_f1_weighted": metrics["f1_weighted"],
                }
            )
        logger.info("Artifacts logged to MLflow")

    logger.info("Done!")


if __name__ == "__main__":
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
        "--run-id",
        type=str,
        default=None,
        help="MLflow run_id (required if --checkpoint is not specified)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: {exp_dir}/validation)",
    )
    args, unknown = parser.parse_known_args()
    args.overrides = unknown

    main(args)
