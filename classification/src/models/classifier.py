from typing import Any

import pytorch_lightning as L
import timm
import torch
import torch.optim as optim
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig
from torchmetrics.classification import MulticlassAccuracy

from builders.loss import build_loss
from builders.optimizer import build_optimizer, build_scheduler


class ImageClassifier(L.LightningModule):
    """画像分類用のPyTorch Lightningモジュール.

    YAML設定からモデル、オプティマイザー、スケジューラーを動的に構築する。
    """

    _config: dict[str, Any]

    def __init__(self, config: dict[str, Any]) -> None:
        """ImageClassifierを初期化する.

        Args:
            config: 設定辞書。以下のキーを含む:
                - model.name: timmのモデル名
                - model.pretrained: 事前学習済み重みを使用するか
                - data.num_classes: クラス数
                - optimizer: オプティマイザー設定
                - scheduler: スケジューラー設定
                - trainer.max_epochs: 最大エポック数
        """
        super().__init__()
        self.save_hyperparameters(config)
        object.__setattr__(self, "_config", config)

        model_config = config.get("model", {})
        data_config = config.get("data", {})

        model_name = model_config.get("name", "efficientnet_b0")
        pretrained = model_config.get("pretrained", True)
        drop_rate = model_config.get("drop_rate", 0.0)
        drop_path_rate = model_config.get("drop_path_rate", 0.0)
        num_classes = data_config.get("num_classes", 1000)

        self.model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

        # 損失関数を構築
        loss_config = config.get("loss", {"type": "cross_entropy"})
        class_counts = data_config.get("class_counts", None)
        self.criterion = build_loss(loss_config, class_counts)

        self.train_acc1 = MulticlassAccuracy(num_classes=num_classes, top_k=1)
        self.val_acc1 = MulticlassAccuracy(num_classes=num_classes, top_k=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """モデルを通した順伝播.

        Args:
            x: 入力テンソル

        Returns:
            モデルの出力テンソル
        """
        return self.model(x)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """訓練ステップ.

        Args:
            batch: (画像, ターゲット)のタプル
            batch_idx: バッチインデックス

        Returns:
            損失を含む辞書
        """
        images, targets = batch
        outputs = self(images)
        loss = self.criterion(outputs, targets)

        self.train_acc1(outputs, targets)

        self.log(
            "loss/train",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        self.log(
            "metrics/top1/train",
            self.train_acc1,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        opt = self.optimizers()
        if opt is not None:
            lr = opt.param_groups[0]["lr"]
            self.log("lr", lr, on_step=True, on_epoch=False, prog_bar=False)

        return {"loss": loss}

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """検証ステップ.

        Args:
            batch: (画像, ターゲット)のタプル
            batch_idx: バッチインデックス
        """
        images, targets = batch
        outputs = self(images)
        loss = self.criterion(outputs, targets)

        self.val_acc1(outputs, targets)

        self.log(
            "loss/val",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "metrics/top1/val",
            self.val_acc1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def configure_optimizers(self) -> optim.Optimizer | OptimizerLRSchedulerConfig:
        """オプティマイザーと学習率スケジューラーを設定する.

        Returns:
            オプティマイザーまたはオプティマイザーとスケジューラーの設定
        """
        optimizer_config = self._config.get("optimizer", {"opt": "adamw"}).copy()
        model_config = self._config.get("model", {})
        scheduler_config = self._config.get("scheduler", None)

        max_steps = self.trainer.estimated_stepping_batches

        lr = model_config.get("lr")
        if lr is None:
            raise ValueError("lr must be specified in model config")
        optimizer_config["lr"] = lr

        optimizer = build_optimizer(optimizer_config, self)

        if scheduler_config is None:
            return optimizer

        min_lr = model_config.get("min_lr", 1e-6)
        scheduler = build_scheduler(
            scheduler_config, optimizer, max_steps, self.trainer.max_epochs, min_lr
        )

        return OptimizerLRSchedulerConfig(
            optimizer=optimizer,
            lr_scheduler={
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        )
