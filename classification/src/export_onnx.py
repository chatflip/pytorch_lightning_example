import argparse
import datetime
import json
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import onnx
import onnxruntime as ort
import onnxsim
import torch
import torch.nn as nn
import yaml
from loguru import logger

from models import ImageClassifier


def _load_config_from_artifact(run_id: str) -> dict[str, Any]:
    """MLflowのartifactからconfig.yamlをダウンロードして読み込む.

    Args:
        run_id: MLflowのrun_id

    Returns:
        復元されたconfig辞書

    Raises:
        FileNotFoundError: config.yamlがartifactに存在しない場合
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            local_path = mlflow.artifacts.download_artifacts(  # type: ignore[possibly-missing-attribute]
                run_id=run_id,
                artifact_path="config.yaml",
                dst_path=tmp_dir,
            )
            config_path = Path(local_path)
            with open(config_path) as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from MLflow artifact: {run_id}/config.yaml")
            return config
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to download config.yaml from MLflow artifacts: {e}"
            ) from e


class ModelWithSoftmax(nn.Module):
    """Softmax付きモデルラッパー.

    推論時に確率（0-1）を出力するためにSoftmaxを追加する。
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """ModelWithSoftmaxを初期化する.

        Args:
            model: 元のPyTorchモデル
        """
        super().__init__()
        self.model = model
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播（Softmax付き）.

        Args:
            x: 入力テンソル

        Returns:
            Softmax適用後の確率テンソル（0-1）
        """
        logits = self.model(x)
        return self.softmax(logits)


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

    model_with_softmax = ModelWithSoftmax(pytorch_model)
    model_with_softmax.eval()

    with torch.no_grad():
        pytorch_outputs = model_with_softmax(dummy_input).numpy()

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
    tracking_uri: str,
    run_id: str,
    checkpoint: str | None = None,
    output_path: str | None = None,
    opset_version: int = 18,
    dynamic_batch: bool = True,
    simplify: bool = False,
    verify: bool = True,
    created_by: str = "unknown",
) -> Path:
    """モデルをONNX形式でエクスポートする.

    MLflowのrunからconfigを取得し、モデルをONNX形式でエクスポートします。

    Args:
        tracking_uri: MLflowのトラッキングURI
        run_id: MLflow run_id（configとチェックポイントの取得に使用）
        checkpoint: チェックポイントファイルのパス（省略時は自動検出）
        output_path: 出力ONNXファイルのパス（省略時は自動生成）
        opset_version: ONNXのopsetバージョン
        dynamic_batch: 動的バッチサイズを有効にするか
        simplify: ONNXモデルを簡略化するか（onnx-simplifier必要）
        verify: エクスポート後にモデルを検証するか
        created_by: メタデータに記録する作成者名

    Returns:
        出力ONNXファイルのパス
    """
    mlflow.set_tracking_uri(tracking_uri)
    config = _load_config_from_artifact(run_id)

    base_output_dir = Path(config.get("output_dir", "./outputs"))
    exp_name = config["exp_name"]

    checkpoint_path, exp_dir = _resolve_checkpoint_path(
        checkpoint, run_id, base_output_dir, exp_name
    )

    input_size = config["model"]["input_size"]
    logger.info(f"Input size: {input_size}x{input_size}")

    output_file = Path(output_path) if output_path else exp_dir / f"{exp_name}.onnx"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output path: {output_file}")

    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    model = ImageClassifier.load_from_checkpoint(checkpoint_path, config=config)
    model.eval()
    model = model.to(torch.device("cpu"))

    dummy_input = torch.randn(1, 3, input_size, input_size)
    logger.info(f"Dummy input shape: {dummy_input.shape}")

    dynamic_axes = (
        {"images": {0: "batch_size"}, "output": {0: "batch_size"}}
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

    _write_onnx_metadata(output_file, config, run_id, created_by)

    if run_id is not None:
        _log_onnx_to_mlflow(run_id, output_file, config)

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

    model_with_softmax = ModelWithSoftmax(model)
    model_with_softmax.eval()

    torch.onnx.export(
        model_with_softmax,
        (dummy_input,),
        str(output_file),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )
    logger.info(f"ONNX model exported to: {output_file} (with Softmax)")


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


def _log_onnx_to_mlflow(run_id: str, onnx_path: Path, config: dict[str, Any]) -> None:
    """ONNXモデルをMLflowのartifactとしてログする.

    Args:
        run_id: MLflowのrun_id
        onnx_path: ONNXファイルのパス
        config: 設定辞書
    """
    logger.info(f"Logging ONNX model to MLflow (run_id={run_id})...")

    logger_config = config.get("logger", {})
    tracking_uri = logger_config.get("tracking_uri", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)  # type: ignore[attr-defined]

    with mlflow.start_run(run_id=run_id):  # type: ignore[attr-defined]
        mlflow.log_artifact(str(onnx_path), artifact_path="onnx")  # type: ignore[attr-defined]

    logger.info(f"ONNX model logged to MLflow artifact: onnx/{onnx_path.name}")


def _get_category_names(config: dict[str, Any]) -> list[str]:
    """データセットからカテゴリ名を取得する.

    Args:
        config: 設定辞書

    Returns:
        カテゴリ名のリスト（ソート済み）

    Raises:
        KeyError: data.dataset_rootが設定に存在しない場合
        FileNotFoundError: trainディレクトリが存在しない場合
    """
    dataset_root = Path(config["data"]["dataset_root"])
    train_dir = dataset_root / "train"

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    categories = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    if not categories:
        raise ValueError(f"No category directories found in: {train_dir}")

    return categories


def _get_best_accuracy_from_mlflow(
    run_id: str | None, config: dict[str, Any]
) -> float | None:
    """MLflowからbest accuracyを取得する.

    Args:
        run_id: MLflowのrun_id
        config: 設定辞書

    Returns:
        best accuracy（取得できない場合はNone）
    """
    if run_id is None:
        return None

    try:
        logger_config = config.get("logger", {})
        tracking_uri = logger_config.get("tracking_uri", "mlruns")
        mlflow.set_tracking_uri(tracking_uri)  # type: ignore[attr-defined]

        run = mlflow.get_run(run_id)  # type: ignore[attr-defined]
        metrics = run.data.metrics
        return metrics["metrics/top1/val"]

    except Exception as e:
        logger.warning(f"Failed to get accuracy from MLflow: {e}")
        return None


def _get_preprocess_info(config: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """前処理情報を設定から取得する.

    Args:
        config: 設定辞書

    Returns:
        (前処理名リスト, 正規化パラメータ辞書)のタプル

    Raises:
        KeyError: augmentation.val.opsが設定に存在しない場合
        ValueError: Normalize設定にmean/stdが存在しない場合
    """
    val_ops = config["augmentation"]["val"]["ops"]

    preprocess_names = []
    normalize_params: dict[str, Any] = {}

    for op in val_ops:
        op_type = op["type"]
        preprocess_names.append(op_type)

        if op_type == "Normalize":
            if "mean" not in op or "std" not in op:
                raise ValueError("Normalize op must have 'mean' and 'std' fields")
            normalize_params = {
                "mean": op["mean"],
                "std": op["std"],
                "max_pixel_value": 255.0,
            }

    if not normalize_params:
        raise ValueError("No Normalize op found in augmentation.val.ops")

    return preprocess_names, normalize_params


def _set_model_metadata(model: onnx.ModelProto, props: dict[str, str]) -> None:
    """ONNXモデルにメタデータを設定する.

    Args:
        model: ONNXモデル
        props: メタデータのkey-valueペア
    """
    while model.metadata_props:
        model.metadata_props.pop()

    for key, value in props.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = value


def _write_onnx_metadata(
    onnx_path: Path,
    config: dict[str, Any],
    run_id: str | None = None,
    created_by: str = "unknown",
) -> None:
    """ONNXモデルにメタデータを書き込む.

    Args:
        onnx_path: ONNXファイルのパス
        config: 設定辞書
        run_id: MLflowのrun_id（精度取得用）
        created_by: 作成者名

    Raises:
        KeyError: 必要な設定が存在しない場合
    """
    logger.info("Writing metadata to ONNX model...")

    model = onnx.load(str(onnx_path))
    props = {p.key: p.value for p in model.metadata_props}
    props["created_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    props["created_by"] = created_by

    category_names = _get_category_names(config)
    props["category_names"] = json.dumps({"images": category_names}, indent=1)

    data_config = config["data"]
    dataset_root = Path(data_config["dataset_root"])
    props["dataset_name"] = dataset_root.name

    accuracy = _get_best_accuracy_from_mlflow(run_id, config)
    if accuracy is not None:
        props["accuracy"] = f"{accuracy:.4f}"
    props["accuracy_detail"] = json.dumps({}, indent=1)

    props["process_type"] = "classification"
    props["process_version"] = "0.0.1"

    preprocess_names, normalize_params = _get_preprocess_info(config)
    props["preprocess"] = json.dumps(preprocess_names, indent=1)
    props["normalize"] = json.dumps(normalize_params, indent=1)
    _set_model_metadata(model, props)

    onnx.save(model, str(onnx_path))
    logger.info("Metadata written successfully!")

    logger.info("Written metadata:")
    for key, value in props.items():
        if value.startswith("{") or value.startswith("["):
            display_value = value.replace("\n", " ")[:50] + "..."
        else:
            display_value = value
        logger.info(f"  {key}: {display_value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export PyTorch Lightning model to ONNX format"
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="sqlite:///mlflow.db",
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="MLflow run_id (required if --checkpoint is not specified)",
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
        "--no-simplify",
        action="store_true",
        help="Simplify ONNX model (requires onnx-simplifier)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip ONNX model verification",
    )
    parser.add_argument(
        "--created-by",
        type=str,
        default="unknown",
        help="Creator name for metadata (default: unknown)",
    )

    args = parser.parse_args()
    export_onnx(
        tracking_uri=args.tracking_uri,
        run_id=args.run_id,
        opset_version=args.opset_version,
        dynamic_batch=not args.no_dynamic_batch,
        simplify=not args.no_simplify,
        verify=not args.no_verify,
        created_by=args.created_by,
    )
