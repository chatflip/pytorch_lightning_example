import math
from typing import Any

import torch.nn as nn
import torch.optim as optim
from timm.optim import create_optimizer_v2
from torch.optim.lr_scheduler import LRScheduler


class FlatCosineScheduler(LRScheduler):
    """DEIMv2スタイルのFlat + Cosine Annealingスケジューラ.

    学習率の推移:
    1. Warmup期間: 0 → base_lr まで線形に増加
    2. Flat期間: base_lr を維持
    3. Cosine期間: base_lr → eta_min までコサインアニーリングで減衰

    Args:
        optimizer: オプティマイザー
        total_steps: 総ステップ数
        warmup_steps: ウォームアップ期間（ステップ数）
        flat_steps: フラット期間（ステップ数、ウォームアップ後）
        eta_min: 最小学習率
        last_epoch: 最後のエポック（再開時に使用）

    Example:
        >>> scheduler = FlatCosineScheduler(
        ...     optimizer,
        ...     total_steps=10000,
        ...     warmup_steps=500,
        ...     flat_steps=2000,
        ...     eta_min=1e-6
        ... )
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        flat_steps: int = 0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """FlatCosineSchedulerを初期化する."""
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.flat_steps = flat_steps
        self.eta_min = eta_min

        # Cosine期間のステップ数
        self.cosine_steps = max(1, total_steps - warmup_steps - flat_steps)

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """現在のステップに対する学習率を計算する."""
        step = self.last_epoch

        if step < self.warmup_steps:
            # Warmup期間: 線形に増加
            alpha = step / max(1, self.warmup_steps)
            return [base_lr * alpha for base_lr in self.base_lrs]

        elif step < self.warmup_steps + self.flat_steps:
            # Flat期間: base_lrを維持
            return list(self.base_lrs)

        else:
            # Cosine期間: コサインアニーリング
            cosine_step = step - self.warmup_steps - self.flat_steps
            progress = cosine_step / self.cosine_steps
            # 0 → π の範囲でコサイン値を計算
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [
                self.eta_min + (base_lr - self.eta_min) * cosine_decay
                for base_lr in self.base_lrs
            ]


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
    max_steps: int | None = None,
    max_epochs: int | None = None,
) -> LRScheduler:
    """YAML設定から学習率スケジューラーを構築する.

    Args:
        config: スケジューラー設定辞書
            例: {"type": "CosineAnnealingLR", "T_max": null, "eta_min": 0.00001}
            FlatCosineSchedulerの場合、warmup_epochs/flat_epochsでエポック指定可能
        optimizer: オプティマイザー
        max_steps: 最大ステップ数（T_maxがnullの場合に使用）
        max_epochs: 最大エポック数（エポック→ステップ変換に使用）

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

    # T_maxがnullの場合はmax_stepsを使用
    if "T_max" in config and config["T_max"] is None:
        if max_steps is None:
            raise ValueError("max_steps is required when T_max is null")
        config["T_max"] = max_steps

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
        "FlatCosineScheduler": FlatCosineScheduler,
    }

    if scheduler_type not in scheduler_map:
        raise ValueError(
            f"Unknown scheduler type: {scheduler_type}. "
            f"Supported: {list(scheduler_map.keys())}"
        )

    scheduler_class = scheduler_map[scheduler_type]

    # FlatCosineSchedulerの場合の処理
    if scheduler_type == "FlatCosineScheduler":
        if max_steps is None:
            raise ValueError("max_steps is required for FlatCosineScheduler")

        # total_stepsを自動設定
        if "total_steps" not in config or config["total_steps"] is None:
            config["total_steps"] = max_steps

        # エポック→ステップ変換
        if max_epochs is not None and max_epochs > 0:
            steps_per_epoch = max_steps // max_epochs

            # warmup_epochs → warmup_steps
            if "warmup_epochs" in config:
                warmup_epochs = config.pop("warmup_epochs")
                config["warmup_steps"] = int(warmup_epochs * steps_per_epoch)

            # flat_epochs → flat_steps
            if "flat_epochs" in config:
                flat_epochs = config.pop("flat_epochs")
                config["flat_steps"] = int(flat_epochs * steps_per_epoch)

    return scheduler_class(optimizer, **config)
