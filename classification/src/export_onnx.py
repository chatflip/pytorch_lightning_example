"""PyTorch LightningモデルをONNX形式でエクスポートするスクリプト.

学習済みチェックポイントをONNX形式に変換し、
推論エンジン（TensorRT、ONNX Runtime等）で利用可能にする。
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import onnxsim
import torch
from loguru import logger

from config import load_config
from models import ImageClassifier


def find_best_checkpoint(checkpoint_dir: Path) -> Path:
    """チェックポイントディレクトリからbest.ckptを見つける.

    Args:
        checkpoint_dir: チェックポイントディレクトリのパス

    Returns:
        best.ckptのパス

    Raises:
        FileNotFoundError: best.ckptが見つからない場合
    """
    best_ckpt = checkpoint_dir / "best.ckpt"
    if best_ckpt.exists():
        return best_ckpt

    raise FileNotFoundError(f"best.ckpt not found in {checkpoint_dir}")


def _resolve_checkpoint_path(
    checkpoint: str | None,
    run_id: str | None,
    base_output_dir: Path,
    exp_name: str,
) -> tuple[Path, Path]:
    """チェックポイントパスと実験ディレクトリを解決する.

    Args:
        checkpoint: チェックポイントファイルのパス
        run_id: MLflow run_id
        base_output_dir: 出力ベースディレクトリ
        exp_name: 実験名

    Returns:
        (チェックポイントパス, 実験ディレクトリ)のタプル

    Raises:
        ValueError: checkpointとrun_idの両方が指定されていない場合
    """
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint)
        exp_dir = checkpoint_path.parent.parent
        logger.info(f"Using checkpoint: {checkpoint_path}")
        return checkpoint_path, exp_dir

    if run_id is not None:
        exp_dir = base_output_dir / exp_name / run_id
        checkpoint_dir = exp_dir / "checkpoints"
        checkpoint_path = find_best_checkpoint(checkpoint_dir)
        logger.info(f"Auto-detected checkpoint: {checkpoint_path}")
        return checkpoint_path, exp_dir

    raise ValueError(
        "Either checkpoint or run_id must be specified. "
        "Use --checkpoint to specify the checkpoint file path, "
        "or --run-id to specify the MLflow run_id."
    )


def _simplify_onnx_model(output_file: Path) -> None:
    """ONNXモデルを簡略化する.

    Args:
        output_file: ONNXファイルのパス
    """
    logger.info("Simplifying ONNX model...")

    onnx_model = onnx.load(str(output_file))
    simplified_model, check = onnxsim.simplify(onnx_model)
    if check:
        onnx.save(simplified_model, str(output_file))
        logger.info("ONNX model simplified successfully!")
    else:
        logger.warning("ONNX model simplification failed, keeping original")


def _print_model_info(onnx_model: onnx.ModelProto, config: dict[str, Any]) -> None:
    """ONNXモデルの情報を表示する.

    Args:
        onnx_model: ONNXモデル
        config: 設定辞書
    """
    graph = onnx_model.graph

    logger.info("=" * 50)
    logger.info("Model Information:")
    for inp in graph.input:
        dims = inp.type.tensor_type.shape.dim
        shape = [dim.dim_value or dim.dim_param for dim in dims]
        logger.info(f"  Input: {inp.name}, shape={shape}")

    for out in graph.output:
        dims = out.type.tensor_type.shape.dim
        shape = [dim.dim_value or dim.dim_param for dim in dims]
        logger.info(f"  Output: {out.name}, shape={shape}")

    num_classes = config.get("data", {}).get("num_classes", "unknown")
    logger.info(f"  Number of classes: {num_classes}")
    logger.info("=" * 50)


def _verify_with_onnxruntime(
    onnx_path: Path,
    dummy_input: torch.Tensor,
    pytorch_model: ImageClassifier,
) -> None:
    """ONNX RuntimeでONNXモデルの推論をテストする.

    Args:
        onnx_path: ONNXファイルのパス
        dummy_input: ダミー入力テンソル
        pytorch_model: 比較用のPyTorchモデル
    """
    logger.info("Testing inference with ONNX Runtime...")

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name
    ort_inputs = {input_name: dummy_input.numpy()}
    ort_outputs = session.run(None, ort_inputs)
    ort_output = np.asarray(ort_outputs[0])

    with torch.no_grad():
        pytorch_outputs = pytorch_model(dummy_input).numpy()

    max_diff = np.abs(ort_output - pytorch_outputs).max()
    mean_diff = np.abs(ort_output - pytorch_outputs).mean()

    logger.info(f"  ONNX Runtime output shape: {ort_output.shape}")
    logger.info(f"  Max difference from PyTorch: {max_diff:.6e}")
    logger.info(f"  Mean difference from PyTorch: {mean_diff:.6e}")

    if max_diff < 1e-4:
        logger.info("  ✓ ONNX Runtime verification passed!")
    else:
        logger.warning(
            f"  ⚠ Large difference detected (max={max_diff:.6e}). "
            "This might be due to floating point precision."
        )


def export_onnx(
    config_path: str,
    checkpoint: str | None = None,
    run_id: str | None = None,
    output_path: str | None = None,
    input_size: int | None = None,
    batch_size: int = 1,
    opset_version: int = 18,
    dynamic_batch: bool = True,
    simplify: bool = False,
    verify: bool = True,
) -> Path:
    """モデルをONNX形式でエクスポートする.

    Args:
        config_path: 設定ファイルのパス
        checkpoint: チェックポイントファイルのパス（省略時は自動検出）
        run_id: MLflow run_id（checkpointが指定されていない場合に必要）
        output_path: 出力ONNXファイルのパス（省略時は自動生成）
        input_size: 入力画像サイズ（省略時は設定から取得）
        batch_size: バッチサイズ（動的バッチの場合はダミー入力用）
        opset_version: ONNXのopsetバージョン
        dynamic_batch: 動的バッチサイズを有効にするか
        simplify: ONNXモデルを簡略化するか（onnx-simplifier必要）
        verify: エクスポート後にモデルを検証するか

    Returns:
        出力ONNXファイルのパス
    """
    logger.info(f"Loading config from: {config_path}")
    config = load_config(config_path)

    base_output_dir = Path(config.get("output_dir", "./outputs"))
    exp_name = config.get("exp_name", "default")

    checkpoint_path, exp_dir = _resolve_checkpoint_path(
        checkpoint, run_id, base_output_dir, exp_name
    )

    model_config = config.get("model", {})
    if input_size is None:
        input_size = model_config.get("input_size", 224)
    logger.info(f"Input size: {input_size}x{input_size}")

    output_file = Path(output_path) if output_path else exp_dir / f"{exp_name}.onnx"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output path: {output_file}")

    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    model = ImageClassifier.load_from_checkpoint(checkpoint_path, config=config)
    model.eval()
    model = model.to(torch.device("cpu"))

    dummy_input = torch.randn(batch_size, 3, input_size, input_size)
    logger.info(f"Dummy input shape: {dummy_input.shape}")

    dynamic_axes = (
        {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        if dynamic_batch
        else None
    )
    if dynamic_batch:
        logger.info("Dynamic batch size enabled")

    _export_to_onnx(model, dummy_input, output_file, opset_version, dynamic_axes)

    if verify:
        _verify_onnx_model(output_file, config)
    if simplify:
        _simplify_onnx_model(output_file)
    if verify:
        _verify_with_onnxruntime(output_file, dummy_input, model)

    logger.info("Done!")
    return output_file


def _export_to_onnx(
    model: ImageClassifier,
    dummy_input: torch.Tensor,
    output_file: Path,
    opset_version: int,
    dynamic_axes: dict[str, dict[int, str]] | None,
) -> None:
    """ONNXファイルへのエクスポートを実行する.

    Args:
        model: PyTorchモデル
        dummy_input: ダミー入力テンソル
        output_file: 出力ファイルパス
        opset_version: ONNXのopsetバージョン
        dynamic_axes: 動的軸の設定
    """
    logger.info(f"Exporting to ONNX (opset_version={opset_version})...")
    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_file),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )
    logger.info(f"ONNX model exported to: {output_file}")


def _verify_onnx_model(output_file: Path, config: dict[str, Any]) -> None:
    """ONNXモデルの検証を行う.

    Args:
        output_file: ONNXファイルパス
        config: 設定辞書
    """
    logger.info("Verifying ONNX model...")
    onnx_model = onnx.load(str(output_file))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model verification passed!")
    _print_model_info(onnx_model, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export PyTorch Lightning model to ONNX format"
    )
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
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output ONNX file path (default: {exp_dir}/{exp_name}.onnx)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=None,
        help="Input image size (default: from config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for dummy input (default: 1)",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=18,
        help="ONNX opset version (default: 18)",
    )
    parser.add_argument(
        "--no-dynamic-batch",
        action="store_true",
        help="Disable dynamic batch size",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Simplify ONNX model (requires onnx-simplifier)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip ONNX model verification",
    )

    args = parser.parse_args()

    export_onnx(
        config_path=args.config,
        checkpoint=args.checkpoint,
        run_id=args.run_id,
        output_path=args.output,
        input_size=args.input_size,
        batch_size=args.batch_size,
        opset_version=args.opset_version,
        dynamic_batch=not args.no_dynamic_batch,
        simplify=args.simplify,
        verify=not args.no_verify,
    )
