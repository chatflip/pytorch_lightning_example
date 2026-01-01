from typing import Any

import torch.nn as nn
import torch.optim as optim
from timm.optim import create_optimizer_v2
from torch.optim.lr_scheduler import LRScheduler


def build_optimizer(config: dict[str, Any], model: nn.Module) -> optim.Optimizer:
    """YAML設定からオプティマイザーを構築する（timm使用）.

    Args:
        config: オプティマイザー設定辞書
            例: {"opt": "adamw", "lr": 0.001, "weight_decay": 0.01}
        model: モデル

    Returns:
        torch.optim.Optimizer オブジェクト

    Example:
        >>> config = {"opt": "adamw", "lr": 0.001, "weight_decay": 0.01}
        >>> optimizer = build_optimizer(config, model)
    """
    config["lr"] = float(config["lr"])
    return create_optimizer_v2(model, **config)


def build_scheduler(
    config: dict[str, Any],
    optimizer: optim.Optimizer,
    max_epochs: int | None = None,
) -> LRScheduler:
    """YAML設定から学習率スケジューラーを構築する.

    Args:
        config: スケジューラー設定辞書
            例: {"type": "CosineAnnealingLR", "T_max": null, "eta_min": 0.00001}
        optimizer: オプティマイザー
        max_epochs: 最大エポック数（T_maxがnullの場合に使用）

    Returns:
        torch.optim.lr_scheduler オブジェクト

    Raises:
        ValueError: 未知のスケジューラータイプの場合

    Example:
        >>> config = {"type": "CosineAnnealingLR", "T_max": 100}
        >>> scheduler = build_scheduler(config, optimizer)
    """
    config = config.copy()
    scheduler_type = config.pop("type")

    # T_maxがnullの場合はmax_epochsを使用
    if "T_max" in config and config["T_max"] is None:
        if max_epochs is None:
            raise ValueError("max_epochs is required when T_max is null")
        config["T_max"] = max_epochs

    scheduler_map = {
        "StepLR": optim.lr_scheduler.StepLR,
        "MultiStepLR": optim.lr_scheduler.MultiStepLR,
        "ExponentialLR": optim.lr_scheduler.ExponentialLR,
        "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR,
        "CosineAnnealingWarmRestarts": optim.lr_scheduler.CosineAnnealingWarmRestarts,
        "OneCycleLR": optim.lr_scheduler.OneCycleLR,
        "ReduceLROnPlateau": optim.lr_scheduler.ReduceLROnPlateau,
        "LinearLR": optim.lr_scheduler.LinearLR,
        "PolynomialLR": optim.lr_scheduler.PolynomialLR,
    }

    if scheduler_type not in scheduler_map:
        raise ValueError(
            f"Unknown scheduler type: {scheduler_type}. "
            f"Supported: {list(scheduler_map.keys())}"
        )

    scheduler_class = scheduler_map[scheduler_type]
    return scheduler_class(optimizer, **config)
