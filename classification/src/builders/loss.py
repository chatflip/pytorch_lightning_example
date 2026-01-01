from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification.

    Reference: https://arxiv.org/abs/1708.02002

    Args:
        gamma: フォーカシングパラメータ。大きいほど難しいサンプルに注力 (default: 2.0)
        alpha: クラス重み。Tensorまたはリスト (default: None)
        label_smoothing: ラベルスムージング係数 (default: 0.0)
        reduction: 出力の削減方法 ('mean', 'sum', 'none')
    """

    gamma: float
    label_smoothing: float
    reduction: Literal["mean", "sum", "none"]
    alpha: torch.Tensor | None

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | list[float] | None = None,
        label_smoothing: float = 0.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        """FocalLossを初期化する."""
        super().__init__()
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "label_smoothing", label_smoothing)
        object.__setattr__(self, "reduction", reduction)

        if alpha is not None:
            if isinstance(alpha, list):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            object.__setattr__(self, "alpha", None)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Focal Lossを計算する.

        Args:
            inputs: モデル出力 (N, C)
            targets: ターゲットラベル (N,)

        Returns:
            計算された損失
        """
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def compute_class_weights(
    class_counts: list[int] | torch.Tensor,
) -> torch.Tensor:
    """クラス数からバランス重みを計算する.

    少数クラスに高い重みを付ける（逆頻度重み付け）。

    Args:
        class_counts: 各クラスのサンプル数

    Returns:
        正規化されたクラス重み
    """
    if isinstance(class_counts, list):
        class_counts = torch.tensor(class_counts, dtype=torch.float32)
    else:
        class_counts = class_counts.float()

    weights = 1.0 / class_counts
    weights = weights / weights.mean()
    return weights


def build_loss(
    config: dict[str, Any],
    class_counts: list[int] | torch.Tensor | None = None,
) -> nn.Module:
    """YAML設定から損失関数を構築する.

    Args:
        config: 損失関数設定辞書
            例: {"type": "cross_entropy", "weight": "balanced"}
        class_counts: 各クラスのサンプル数（weight="balanced"の場合に必要）

    Returns:
        損失関数

    Raises:
        ValueError: 未知の損失関数タイプ、または必要なパラメータが不足している場合

    Example:
        >>> config = {"type": "focal", "gamma": 2.0, "alpha": "balanced"}
        >>> loss_fn = build_loss(config, class_counts=[100, 50, 10])
    """
    config = config.copy()
    loss_type = config.pop("type", "cross_entropy")

    weight = config.pop("weight", None)
    alpha = config.pop("alpha", None)

    if weight == "balanced" or alpha == "balanced":
        if class_counts is None:
            raise ValueError(
                "class_counts is required when weight/alpha='balanced'. "
                "Set class_counts in data config or provide explicit weights."
            )
        computed_weights = compute_class_weights(class_counts)
        if weight == "balanced":
            weight = computed_weights
        if alpha == "balanced":
            alpha = computed_weights

    if loss_type == "cross_entropy":
        label_smoothing = config.pop("label_smoothing", 0.0)
        if isinstance(weight, list):
            weight = torch.tensor(weight, dtype=torch.float32)
        loss_fn = nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=label_smoothing,
        )
    elif loss_type == "focal":
        gamma = config.pop("gamma", 2.0)
        label_smoothing = config.pop("label_smoothing", 0.0)
        loss_fn = FocalLoss(
            gamma=gamma,
            alpha=alpha,
            label_smoothing=label_smoothing,
        )
    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. Supported: ['cross_entropy', 'focal']"
        )

    return loss_fn
