import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """辞書を再帰的にマージする.

    Args:
        base: ベースとなる辞書
        override: オーバーライドする辞書

    Returns:
        マージされた新しい辞書
    """
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _expand_env_vars(config: Any) -> Any:
    """設定内の環境変数を展開する.

    ${ENV_VAR} 形式の環境変数を展開する。

    Args:
        config: 設定オブジェクト（辞書、リスト、または文字列）

    Returns:
        環境変数が展開された設定
    """
    if isinstance(config, dict):
        return {k: _expand_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_expand_env_vars(item) for item in config]
    elif isinstance(config, str):
        pattern = r"\$\{([^}]+)\}"
        matches = re.findall(pattern, config)
        result = config
        for match in matches:
            env_value = os.environ.get(match, "")
            result = result.replace(f"${{{match}}}", env_value)
        return result
    else:
        return config


def _load_yaml_file(filepath: str | Path) -> dict[str, Any]:
    """YAMLファイルを読み込む.

    Args:
        filepath: YAMLファイルのパス

    Returns:
        パースされた辞書

    Raises:
        FileNotFoundError: ファイルが存在しない場合
        yaml.YAMLError: YAMLパースエラーの場合
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config if config is not None else {}


def _resolve_base_path(base_path: str, current_dir: Path) -> Path:
    """ベースパスを解決する.

    Args:
        base_path: __base__で指定されたパス（相対パス）
        current_dir: 現在の設定ファイルがあるディレクトリ

    Returns:
        解決された絶対パス
    """
    resolved = (current_dir / base_path).resolve()
    return resolved


def _load_config_with_inheritance(
    filepath: str | Path, loaded_files: set[str] | None = None
) -> dict[str, Any]:
    """継承を処理しながら設定を読み込む.

    Args:
        filepath: 設定ファイルのパス
        loaded_files: 既に読み込んだファイルのセット（循環参照防止）

    Returns:
        継承がマージされた設定辞書
    """
    filepath = Path(filepath).resolve()

    if loaded_files is None:
        loaded_files = set()

    filepath_str = str(filepath)
    if filepath_str in loaded_files:
        raise ValueError(f"Circular inheritance detected: {filepath}")
    loaded_files.add(filepath_str)

    config = _load_yaml_file(filepath)
    current_dir = filepath.parent

    # __base__ キーがあれば継承処理
    if "__base__" in config:
        base_paths = config.pop("__base__")

        # 文字列の場合はリストに変換
        if isinstance(base_paths, str):
            base_paths = [base_paths]

        # ベース設定をマージ
        merged_base: dict[str, Any] = {}
        for base_path in base_paths:
            resolved_path = _resolve_base_path(base_path, current_dir)
            base_config = _load_config_with_inheritance(
                resolved_path, loaded_files.copy()
            )
            merged_base = _deep_merge(merged_base, base_config)

        # ベース設定に現在の設定をマージ
        config = _deep_merge(merged_base, config)

    return config


def load_config(filepath: str | Path) -> dict[str, Any]:
    """設定ファイルを読み込む.

    YAML継承と環境変数の展開を処理する。

    Args:
        filepath: 設定ファイルのパス

    Returns:
        完全にマージされた設定辞書

    Example:
        >>> config = load_config("config/experiments/food101.yaml")
        >>> print(config["model"]["name"])
        'efficientnet_b0'
    """
    config = _load_config_with_inheritance(filepath)
    config = _expand_env_vars(config)
    return config


def save_config(config: dict[str, Any], filepath: str | Path) -> None:
    """設定を保存する.

    Args:
        config: 保存する設定辞書
        filepath: 保存先のパス
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(
            config, f, default_flow_style=False, allow_unicode=True, sort_keys=False
        )
