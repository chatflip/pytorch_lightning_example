"""画像分類用のPyTorch Lightningモジュール.

timmモデルとYAML設定に基づく動的オプティマイザー設定をサポート。
torchmetricsを使用した精度計算。
"""

from typing import Any

import pytorch_lightning as L
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig
from torchmetrics.classification import MulticlassAccuracy

from builders.optimizer import build_optimizer, build_scheduler


class ImageClassifier(L.LightningModule):
    """画像分類用のPyTorch Lightningモジュール.

    YAML設定からモデル、オプティマイザー、スケジューラーを動的に構築する。
    """

    # クラス変数として設定を保持するための型アノテーション
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
        # __setattr__を回避してobject経由で設定
        object.__setattr__(self, "_config", config)

        # モデル設定
        model_config = config.get("model", {})
        data_config = config.get("data", {})

        model_name = model_config.get("name", "efficientnet_b0")
        pretrained = model_config.get("pretrained", True)
        drop_rate = model_config.get("drop_rate", 0.0)
        drop_path_rate = model_config.get("drop_path_rate", 0.0)
        num_classes = data_config.get("num_classes", 1000)

        # timmモデルを作成
        self.model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

        # 損失関数
        self.criterion = nn.CrossEntropyLoss()

        # torchmetricsを使用したメトリクス
        self.train_acc1 = MulticlassAccuracy(num_classes=num_classes, top_k=1)
        self.train_acc5 = MulticlassAccuracy(num_classes=num_classes, top_k=5)
        self.val_acc1 = MulticlassAccuracy(num_classes=num_classes, top_k=1)
        self.val_acc5 = MulticlassAccuracy(num_classes=num_classes, top_k=5)
        self.test_acc1 = MulticlassAccuracy(num_classes=num_classes, top_k=1)
        self.test_acc5 = MulticlassAccuracy(num_classes=num_classes, top_k=5)

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

        # メトリクスを更新
        self.train_acc1(outputs, targets)
        self.train_acc5(outputs, targets)

        # ログ
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_acc1",
            self.train_acc1,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("train_acc5", self.train_acc5, on_step=False, on_epoch=True)

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

        # メトリクスを更新
        self.val_acc1(outputs, targets)
        self.val_acc5(outputs, targets)

        # ログ
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc1", self.val_acc1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc5", self.val_acc5, on_step=False, on_epoch=True)

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """テストステップ.

        Args:
            batch: (画像, ターゲット)のタプル
            batch_idx: バッチインデックス
        """
        images, targets = batch
        outputs = self(images)
        loss = self.criterion(outputs, targets)

        # メトリクスを更新
        self.test_acc1(outputs, targets)
        self.test_acc5(outputs, targets)

        # ログ
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc1", self.test_acc1, on_step=False, on_epoch=True)
        self.log("test_acc5", self.test_acc5, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> optim.Optimizer | OptimizerLRSchedulerConfig:
        """オプティマイザーと学習率スケジューラーを設定する.

        Returns:
            オプティマイザーまたはオプティマイザーとスケジューラーの設定
        """
        optimizer_config = self._config.get("optimizer", {"type": "AdamW", "lr": 0.001})
        scheduler_config = self._config.get("scheduler", None)
        trainer_config = self._config.get("trainer", {})
        max_epochs = trainer_config.get("max_epochs", 100)

        # オプティマイザーを構築
        optimizer = build_optimizer(optimizer_config, self.parameters())

        if scheduler_config is None:
            return optimizer

        # スケジューラーを構築
        scheduler = build_scheduler(scheduler_config, optimizer, max_epochs)

        return OptimizerLRSchedulerConfig(
            optimizer=optimizer,
            lr_scheduler={
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        )
